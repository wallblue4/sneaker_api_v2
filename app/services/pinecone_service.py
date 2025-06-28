# app/services/pinecone_service.py
import pinecone
from typing import List, Dict, Optional, Any
import logging
import asyncio
from app.core.config import settings

logger = logging.getLogger(__name__)

class PineconeService:
    """Servicio para búsquedas vectoriales en Pinecone"""
    
    def __init__(self):
        self.api_key = settings.PINECONE_API_KEY
        self.index_name = settings.PINECONE_INDEX_NAME
        self.pc = None
        self.index = None
        
        if self.api_key:
            try:
                self.pc = pinecone.Pinecone(api_key=self.api_key)
                self.index = self.pc.Index(self.index_name)
                logger.info(f"✅ Pinecone inicializado para índice '{self.index_name}'")
            except Exception as e:
                logger.error(f"❌ Error inicializando Pinecone: {e}")
        else:
            logger.warning("⚠️ PINECONE_API_KEY no configurada")
    
    async def search_similar(
        self, 
        query_embedding: List[float], 
        top_k: int = 5,
        filter_dict: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Buscar vectores similares en Pinecone
        
        Args:
            query_embedding: Vector de consulta
            top_k: Número de resultados
            filter_dict: Filtros de metadata
            
        Returns:
            Lista de resultados con metadata
        """
        if not self.index:
            logger.error("❌ Pinecone no inicializado")
            return []
        
        try:
            # Limitar top_k para evitar costos excesivos
            top_k = min(top_k, settings.MAX_TOP_K)
            
            query_params = {
                "vector": query_embedding,
                "top_k": top_k,
                "include_metadata": True
            }
            
            if filter_dict:
                query_params["filter"] = filter_dict
            
            logger.info(f"🔍 Buscando {top_k} similares en Pinecone...")
            
            # Ejecutar query en thread pool para no bloquear
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None, 
                lambda: self.index.query(**query_params)
            )
            
            # Formatear resultados matching tu estructura de datos
            formatted_results = []
            for i, match in enumerate(results.matches):
                metadata = match.metadata or {}
                
                result = {
                    "rank": i + 1,
                    "id": match.id,
                    "similarity_score": float(match.score),
                    "confidence_percentage": max(0.0, min(100.0, float(match.score) * 100)),
                    # Datos del sneaker (matching tu migración)
                    "model_name": metadata.get("model_name", "Unknown"),
                    "brand": metadata.get("brand", "Unknown"),
                    "color": metadata.get("color", "Unknown"),
                    "size": metadata.get("size", "Unknown"),
                    "price": float(metadata.get("price", 0.0)),
                    "description": metadata.get("description", ""),
                    "image_path": metadata.get("image_path", ""),
                    # Metadata adicional
                    "original_db_id": metadata.get("original_db_id"),
                    "embedding_index": metadata.get("embedding_index")
                }
                formatted_results.append(result)
            
            logger.info(f"✅ Encontrados {len(formatted_results)} resultados")
            return formatted_results
            
        except Exception as e:
            logger.error(f"❌ Error buscando en Pinecone: {e}")
            return []
    
    async def search_by_brand(
        self, 
        query_embedding: List[float], 
        brand: str, 
        top_k: int = 5
    ) -> List[Dict]:
        """Buscar solo en una marca específica"""
        filter_dict = {"brand": {"$eq": brand}}
        return await self.search_similar(query_embedding, top_k, filter_dict)
    
    async def search_by_price_range(
        self, 
        query_embedding: List[float], 
        min_price: float, 
        max_price: float, 
        top_k: int = 5
    ) -> List[Dict]:
        """Buscar en un rango de precios"""
        filter_dict = {
            "price": {
                "$gte": min_price,
                "$lte": max_price
            }
        }
        return await self.search_similar(query_embedding, top_k, filter_dict)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del índice"""
        if not self.index:
            return {"error": "Pinecone no inicializado"}
        
        try:
            loop = asyncio.get_event_loop()
            stats = await loop.run_in_executor(
                None, 
                self.index.describe_index_stats
            )
            
            return {
                "total_vectors": stats.total_vector_count,
                "dimension": stats.dimension,
                "index_fullness": stats.index_fullness,
                "namespaces": dict(stats.namespaces) if stats.namespaces else {}
            }
            
        except Exception as e:
            logger.error(f"❌ Error obteniendo stats: {e}")
            return {"error": str(e)}
    
    async def health_check(self) -> bool:
        """Verificar conexión con Pinecone"""
        if not self.index:
            return False
            
        try:
            stats = await self.get_stats()
            return "total_vectors" in stats and "error" not in stats
        except:
            return False
    
    async def get_service_info(self) -> Dict[str, Any]:
        """Información del servicio Pinecone"""
        stats = await self.get_stats()
        return {
            "service": "Pinecone Vector Search",
            "index_name": self.index_name,
            "api_configured": bool(self.api_key),
            "stats": stats
        }