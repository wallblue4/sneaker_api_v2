# app/api/routes/classification.py
from fastapi import APIRouter, UploadFile, HTTPException, Depends, File, Query
from typing import List, Optional, Dict
import time
import logging
from datetime import datetime

from app.core.dependencies import get_services
from app.models.requests import SearchByTextRequest
from app.models.responses import (
    ClassificationResponse, 
    SearchResponse,
    SneakerResult, 
    ImageInfo,
    ConfidenceLevel
)
from app.utils.image_utils import validate_and_process_image, get_confidence_level
from app.core.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)


# app/api/routes/classification.py

# app/api/routes/classification.py

async def search_unique_models_optimized(
    pinecone_service,
    embedding: List[float],
    target_unique_models: int,
    filter_dict: Optional[Dict] = None
) -> List[Dict]:
    """
    Búsqueda optimizada para encontrar modelos únicos
    """
    unique_models = {}
    search_size = target_unique_models * 3
    max_iterations = 3
    max_search_limit = 100
    
    logger.info(f"🎯 Búsqueda optimizada: objetivo={target_unique_models} modelos únicos")
    
    for iteration in range(max_iterations):
        current_search_size = min(search_size, max_search_limit)
        
        logger.info(f"🔍 Iteración {iteration + 1}: buscando {current_search_size} vectores")
        
        # 🎯 CORRECCIÓN: Usar el nombre correcto del parámetro
        search_results = await pinecone_service.search_similar(
            query_embedding=embedding,     # ✅ CORRECTO: query_embedding
            top_k=current_search_size,     # ✅ CORRECTO: top_k
            filter_dict=filter_dict        # ✅ CORRECTO: filter_dict
        )
        
        if not search_results:
            logger.warning("⚠️ No se encontraron resultados en Pinecone")
            break
        
        # Procesar resultados y agrupar por modelo
        initial_count = len(unique_models)
        for result in search_results:
            model_name = result.get("model_name")
            
            if not model_name:
                continue
                
            # Si es un modelo nuevo O tiene mejor score que el existente
            if (model_name not in unique_models or 
                result["similarity_score"] > unique_models[model_name]["similarity_score"]):
                unique_models[model_name] = result
        
        new_models_found = len(unique_models) - initial_count
        logger.info(f"📊 Iteración {iteration + 1}: {len(unique_models)} modelos únicos (+{new_models_found} nuevos)")
        
        # Si ya tenemos suficientes modelos únicos, terminar
        if len(unique_models) >= target_unique_models:
            logger.info(f"✅ Objetivo alcanzado: {len(unique_models)} >= {target_unique_models}")
            break
        
        # Si no encontramos nuevos modelos, incrementar búsqueda
        if new_models_found == 0:
            search_size = min(search_size * 2, max_search_limit)
            logger.info(f"🔄 Expandiendo búsqueda a {search_size}")
            
            if current_search_size >= max_search_limit:
                logger.warning(f"⚠️ Alcanzado límite máximo de búsqueda: {max_search_limit}")
                break
        else:
            search_size = min(search_size + 20, max_search_limit)
    
    # Ordenar por similitud y tomar top_k
    sorted_models = sorted(
        unique_models.values(),
        key=lambda x: x["similarity_score"],
        reverse=True
    )[:target_unique_models]
    
    logger.info(f"🎉 Búsqueda completada: {len(sorted_models)} modelos únicos finales")
    
    return sorted_models

@router.post("/classify", response_model=ClassificationResponse)
async def classify_sneaker_by_image(
    image: UploadFile = File(..., description="Imagen de sneaker a clasificar"),
    top_k: int = Query(5, ge=1, le=20, description="Número de modelos únicos a retornar"),
    brand: Optional[str] = Query(None, description="Filtrar por marca específica"),
    min_price: Optional[float] = Query(None, ge=0, description="Precio mínimo"),
    max_price: Optional[float] = Query(None, ge=0, description="Precio máximo"),
    services = Depends(get_services)
):
    """
    Clasificar sneaker garantizando modelos únicos diferentes
    
    - **image**: Archivo de imagen (JPEG, PNG, etc.)
    - **top_k**: Número de MODELOS ÚNICOS similares (1-20)
    - **brand**: Filtrar solo por una marca específica
    - **min_price/max_price**: Filtrar por rango de precios
    """
    start_time = time.time()
    embedding_service, pinecone_service = services
    
    try:
        # 1. Validar y procesar imagen
        logger.info(f"📷 Procesando imagen: {image.filename}")
        image_data, image_info = await validate_and_process_image(image)
        
        # 2. Generar embedding
        logger.info("🔄 Generando embedding con Jina AI...")
        embedding = await embedding_service.get_image_embedding(image_data)
        
        # 3. Preparar filtros
        filter_dict = {}
        filters_applied = {}
        
        if brand:
            filter_dict["brand"] = {"$eq": brand}
            filters_applied["brand"] = brand
            
        if min_price is not None or max_price is not None:
            price_filter = {}
            if min_price is not None:
                price_filter["$gte"] = min_price
                filters_applied["min_price"] = min_price
            if max_price is not None:
                price_filter["$lte"] = max_price
                filters_applied["max_price"] = max_price
            filter_dict["price"] = price_filter
        
        # 4. 🎯 BÚSQUEDA OPTIMIZADA CON PARÁMETROS CORRECTOS
        logger.info(f"🔍 Buscando {top_k} modelos únicos en Pinecone...")
        unique_results = await search_unique_models_optimized(
            pinecone_service=pinecone_service,
            embedding=embedding,
            target_unique_models=top_k,
            filter_dict=filter_dict if filter_dict else None
        )
        
        # 5. Formatear resultados
        results = []
        for i, result in enumerate(unique_results):
            sneaker_result = SneakerResult(
                rank=i + 1,
                similarity_score=result["similarity_score"],
                confidence_percentage=result["confidence_percentage"],
                confidence_level=get_confidence_level(result["similarity_score"]),
                model_name=result["model_name"],
                brand=result["brand"],
                color=result.get("color"),
                size=result.get("size"),
                price=result.get("price"),
                description=result.get("description"),
                image_path=result.get("image_path"),
                original_db_id=result.get("original_db_id")
            )
            results.append(sneaker_result)
        
        processing_time = (time.time() - start_time) * 1000
        
        logger.info(f"✅ Clasificación completada en {processing_time:.2f}ms - {len(results)} modelos únicos")
        
        return ClassificationResponse(
            processing_time_ms=processing_time,
            results=results,
            total_matches_found=len(results),
            query_info=ImageInfo(**image_info),
            model_info={
                "embedding_service": "Jina AI",
                "model": "jina-clip-v2",
                "dimension": settings.EMBEDDING_DIMENSION,
                "filters_applied": filters_applied,
                "search_strategy": "unique_models_optimized"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error clasificando imagen: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error interno procesando imagen: {str(e)}"
        )

@router.post("/search-text", response_model=SearchResponse)
async def search_sneakers_by_text(
   request: SearchByTextRequest,
   services = Depends(get_services)
):
    """Búsqueda por texto también con modelos únicos"""
    start_time = time.time()
    embedding_service, pinecone_service = services
    
    try:
        # 1. Generar embedding del texto
        logger.info(f"🔄 Generando embedding para: '{request.query}'")
        embedding = await embedding_service.get_text_embedding(request.query)
        
        # 2. Preparar filtros
        filter_dict = {}
        filters_applied = {}
        
        if request.brand:
            filter_dict["brand"] = {"$eq": request.brand}
            filters_applied["brand"] = request.brand
            
        if request.min_price is not None or request.max_price is not None:
            price_filter = {}
            if request.min_price is not None:
                price_filter["$gte"] = request.min_price
                filters_applied["min_price"] = request.min_price
            if request.max_price is not None:
                price_filter["$lte"] = request.max_price
                filters_applied["max_price"] = request.max_price
            filter_dict["price"] = price_filter
        
        # 3. 🎯 USAR LA MISMA LÓGICA OPTIMIZADA CON PARÁMETROS CORRECTOS
        unique_results = await search_unique_models_optimized(
            pinecone_service=pinecone_service,
            embedding=embedding,
            target_unique_models=request.top_k,
            filter_dict=filter_dict if filter_dict else None
        )
        
        # 4. Formatear resultados
        results = []
        for i, result in enumerate(unique_results):
            sneaker_result = SneakerResult(
                rank=i + 1,
                similarity_score=result["similarity_score"],
                confidence_percentage=result["confidence_percentage"],
                confidence_level=get_confidence_level(result["similarity_score"]),
                model_name=result["model_name"],
                brand=result["brand"],
                color=result.get("color"),
                size=result.get("size"),
                price=result.get("price"),
                description=result.get("description"),
                image_path=result.get("image_path"),
                original_db_id=result.get("original_db_id")
            )
            results.append(sneaker_result)
        
        processing_time = (time.time() - start_time) * 1000
        
        return SearchResponse(
            processing_time_ms=processing_time,
            query=request.query,
            results=results,
            total_matches_found=len(results),
            filters_applied=filters_applied
        )
        
    except Exception as e:
        logger.error(f"❌ Error en búsqueda por texto: {e}")
        raise HTTPException(500, f"Error en búsqueda: {str(e)}")

@router.get("/brands")
async def get_available_brands(
   pinecone_service = Depends(get_services)
):
   """Obtener lista de marcas disponibles en la base de datos"""
   try:
       # Esta es una operación que requeriría una consulta especial a Pinecone
       # Por simplicidad, devolvemos las marcas más comunes
       common_brands = [
           "Nike", "Adidas", "Jordan", "Puma", "New Balance", 
           "Converse", "Vans", "Reebok", "ASICS", "Under Armour"
       ]
       
       return {
           "success": True,
           "brands": common_brands,
           "total": len(common_brands),
           "note": "Lista de marcas más comunes - para lista completa usar stats endpoint"
       }
       
   except Exception as e:
       raise HTTPException(500, f"Error obteniendo marcas: {str(e)}")

# app/api/routes/classification.py - Arreglar el endpoint de stats

# ... todo el código anterior igual ...

@router.get("/stats")
async def get_database_stats(services = Depends(get_services)):
    """Obtener estadísticas de la base de datos vectorial"""
    try:
        _, pinecone_service = services
        raw_stats = await pinecone_service.get_stats()
        
        # Limpiar stats para evitar objetos no serializables
        clean_stats = {}
        for key, value in raw_stats.items():
            if isinstance(value, (str, int, float, bool, type(None))):
                clean_stats[key] = value
            elif isinstance(value, dict):
                # Limpiar diccionarios anidados
                clean_dict = {}
                for k, v in value.items():
                    if isinstance(v, (str, int, float, bool, type(None))):
                        clean_dict[k] = v
                    else:
                        clean_dict[k] = str(v)  # Convertir a string si no es serializable
                clean_stats[key] = clean_dict
            else:
                clean_stats[key] = str(value)  # Fallback a string
        
        return {
            "success": True,
            "database_stats": clean_stats,
            "timestamp": time.time(),
            "summary": {
                "total_vectors": clean_stats.get("total_vectors", 0),
                "dimension": clean_stats.get("dimension", 0),
                "index_fullness_percent": round(clean_stats.get("index_fullness", 0.0) * 100, 2)
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Error obteniendo estadísticas: {e}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": time.time()
        }