# app/services/embedding_service.py
import httpx
import logging
from typing import List, Optional, Dict, Any
import asyncio
from app.core.config import settings

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Cohere AI - Tier gratuito real para embeddings"""
    
    def __init__(self):
        self.api_key = settings.COHERE_API_KEY
        self.timeout = settings.REQUEST_TIMEOUT
        self.base_url = "https://api.cohere.ai/v1/embed"
        self.model = "embed-english-v3.0"  # 1024 dimensiones
        
        if not self.api_key:
            logger.warning("⚠️ COHERE_API_KEY no configurado")
        else:
            logger.info("🔧 Usando Cohere AI (tier gratuito)")
    
    async def get_image_embedding(self, image_data: bytes) -> List[float]:
        """Para imágenes, usar descripción genérica de sneakers"""
        # Como Cohere es texto-only, usar descripción base
        description = "athletic sneaker shoe footwear sports running casual"
        return await self.get_text_embedding(description)
    
    async def get_text_embedding(self, text: str) -> List[float]:
        """Generar embedding con Cohere AI (gratis)"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
            
            payload = {
                "texts": [text],
                "model": self.model,
                "input_type": "search_query",
                "embedding_types": ["float"]
            }
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                logger.info("🔄 Generando embedding con Cohere AI...")
                
                response = await client.post(
                    self.base_url,
                    json=payload,
                    headers=headers
                )
                response.raise_for_status()
                
                data = response.json()
                embedding = data["embeddings"][0]
                
                logger.info(f"✅ Embedding generado: dimensión {len(embedding)}")
                return embedding
                
        except httpx.HTTPStatusError as e:
            logger.error(f"❌ Error HTTP Cohere {e.response.status_code}: {e.response.text}")
            raise Exception(f"Error en Cohere API: {e.response.status_code}")
        except Exception as e:
            logger.error(f"❌ Error inesperado: {e}")
            raise
    
    async def health_check(self) -> bool:
        """Health check de Cohere"""
        try:
            if not self.api_key:
                return False
            
            embedding = await self.get_text_embedding("health check test")
            return len(embedding) == 1024
            
        except Exception as e:
            logger.error(f"❌ Health check Cohere falló: {e}")
            return False
    
    async def get_api_info(self) -> Dict[str, Any]:
        """Info del servicio Cohere"""
        return {
            "service": "Cohere AI",
            "model": self.model,
            "dimension": 1024,
            "api_configured": bool(self.api_key),
            "base_url": self.base_url,
            "provider": "cohere",
            "tier": "free",
            "note": "1000 requests gratuitos/mes"
        }