# app/services/search_optimizer.py 

from typing import List, Dict, Optional
import logging
from app.core.config import settings

logger = logging.getLogger(__name__)

async def search_unique_models_optimized(
    pinecone_service,
    embedding: List[float],
    target_unique_models: int,
    filter_dict: Optional[Dict] = None,
    max_iterations: int = 5
) -> List[Dict]:
    """
    Búsqueda optimizada para encontrar modelos únicos sin cargar todos los vectores
    
    Estrategia:
    1. Buscar incrementalmente hasta encontrar suficientes modelos únicos
    2. Usar multiplicadores inteligentes
    3. Límite de iteraciones para evitar loops infinitos
    """
    
    unique_models = {}  # {model_name: best_result}
    search_size = target_unique_models * settings.CLASSIFICATION_SEARCH_MULTIPLIER  # Empezar con 3x
    max_search_size = settings.CLASSIFICATION_MAX_SEARCH
    iteration = 0
    
    logger.info(f"🎯 Búsqueda optimizada: objetivo={target_unique_models} modelos únicos")
    
    while len(unique_models) < target_unique_models and iteration < max_iterations:
        iteration += 1
        current_search_size = min(search_size, max_search_size)
        
        logger.info(f"🔄 Iteración {iteration}: buscando {current_search_size} vectores")
        
        # Buscar vectores en Pinecone
        search_results = await pinecone_service.search_similar(
            embedding=embedding,
            top_k=current_search_size,
            filter_dict=filter_dict
        )
        
        if not search_results:
            logger.warning("⚠️ No se encontraron resultados en Pinecone")
            break
        
        # Procesar resultados y agrupar por modelo
        new_models_found = 0
        for result in search_results:
            model_name = result.get("model_name")
            
            if not model_name:
                continue
                
            # Si es un modelo nuevo O tiene mejor score que el existente
            if (model_name not in unique_models or 
                result["similarity_score"] > unique_models[model_name]["similarity_score"]):
                
                if model_name not in unique_models:
                    new_models_found += 1
                    
                unique_models[model_name] = result
        
        logger.info(f"📊 Iteración {iteration}: {len(unique_models)} modelos únicos encontrados (+{new_models_found} nuevos)")
        
        # Si ya tenemos suficientes modelos únicos, terminar
        if len(unique_models) >= target_unique_models:
            logger.info(f"✅ Objetivo alcanzado: {len(unique_models)} >= {target_unique_models}")
            break
        
        # Si no encontramos nuevos modelos, incrementar búsqueda
        if new_models_found == 0:
            search_size = min(search_size * 2, max_search_size)
            logger.info(f"🔄 Expandiendo búsqueda a {search_size}")
            
            # Si ya llegamos al máximo, terminar
            if current_search_size >= max_search_size:
                logger.warning(f"⚠️ Alcanzado límite máximo de búsqueda: {max_search_size}")
                break
        else:
            # Incremento moderado si encontramos algunos modelos nuevos
            search_size = min(search_size + settings.CLASSIFICATION_BATCH_SIZE, max_search_size)
    
    # Ordenar por similitud y tomar top_k
    sorted_models = sorted(
        unique_models.values(),
        key=lambda x: x["similarity_score"],
        reverse=True
    )[:target_unique_models]
    
    logger.info(f"🎉 Búsqueda completada: {len(sorted_models)} modelos únicos finales en {iteration} iteraciones")
    
    return sorted_models


async def search_unique_models_fallback(
    pinecone_service,
    embedding: List[float],
    target_unique_models: int,
    filter_dict: Optional[Dict] = None
) -> List[Dict]:
    """
    Fallback: si la búsqueda optimizada no encuentra suficientes modelos,
    hacer una búsqueda más amplia de una sola vez
    """
    logger.info("🔄 Ejecutando búsqueda fallback más amplia...")
    
    # Buscar una cantidad grande de vectores de una vez
    large_search_size = min(500, settings.CLASSIFICATION_MAX_SEARCH * 2)
    
    search_results = await pinecone_service.search_similar(
        embedding=embedding,
        top_k=large_search_size,
        filter_dict=filter_dict
    )
    
    if not search_results:
        return []
    
    # Agrupar por modelo (lógica igual que en Colab)
    unique_models = {}
    for result in search_results:
        model_name = result.get("model_name")
        
        if not model_name:
            continue
            
        if (model_name not in unique_models or 
            result["similarity_score"] > unique_models[model_name]["similarity_score"]):
            unique_models[model_name] = result
    
    # Retornar top_k modelos únicos
    sorted_models = sorted(
        unique_models.values(),
        key=lambda x: x["similarity_score"],
        reverse=True
    )[:target_unique_models]
    
    logger.info(f"🎯 Fallback completado: {len(sorted_models)} modelos únicos")
    return sorted_models