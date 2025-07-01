# app/services/embedding_service.py
import httpx
import logging
from typing import List, Optional, Dict, Any
import asyncio
import json
from app.core.config import settings

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Cohere AI con debug mejorado"""
    
    def __init__(self):
        self.api_key = settings.COHERE_API_KEY
        self.timeout = settings.REQUEST_TIMEOUT
        self.base_url = "https://api.cohere.ai/v1/embed"
        self.model = "embed-english-v3.0"
        
        if not self.api_key:
            logger.warning("⚠️ COHERE_API_KEY no configurado")
        else:
            logger.info("🔧 Usando Cohere AI (tier gratuito)")
    
    async def get_image_embedding(self, image_data: bytes) -> List[float]:
        """Para imágenes, usar descripción genérica"""
        description = "athletic sneaker shoe footwear sports running casual"
        return await self.get_text_embedding(description)
    
    async def get_text_embedding(self, text: str) -> List[float]:
        """Generar embedding con Cohere AI con debug completo"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
            
            payload = {
                "texts": [text],
                "model": self.model,
                "input_type": "search_query"
            }
            
            logger.info(f"🔄 Enviando request a Cohere: {json.dumps(payload, indent=2)}")
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    self.base_url,
                    json=payload,
                    headers=headers
                )
                
                logger.info(f"📥 Response status: {response.status_code}")
                logger.info(f"📥 Response headers: {dict(response.headers)}")
                
                response_text = response.text
                logger.info(f"📥 Response body: {response_text[:500]}...")  # Primeros 500 chars
                
                if response.status_code != 200:
                    logger.error(f"❌ HTTP Error {response.status_code}: {response_text}")
                    raise Exception(f"Cohere API error {response.status_code}: {response_text}")
                
                try:
                    data = response.json()
                    logger.info(f"📊 Parsed JSON keys: {list(data.keys())}")
                    
                    if "embeddings" in data:
                        embeddings = data["embeddings"]
                        logger.info(f"📊 Embeddings type: {type(embeddings)}, length: {len(embeddings)}")
                        
                        if embeddings and len(embeddings) > 0:
                            embedding = embeddings[0]
                            logger.info(f"📊 First embedding type: {type(embedding)}, length: {len(embedding) if hasattr(embedding, '__len__') else 'N/A'}")
                            
                            if isinstance(embedding, list) and len(embedding) > 0:
                                logger.info(f"✅ Embedding generado exitosamente: {len(embedding)} dimensiones")
                                logger.info(f"📊 Primeros 5 valores: {embedding[:5]}")
                                return embedding
                            else:
                                logger.error(f"❌ Embedding vacío o formato inválido: {embedding}")
                                raise Exception(f"Embedding inválido: {type(embedding)}")
                        else:
                            logger.error(f"❌ Lista de embeddings vacía: {embeddings}")
                            raise Exception("Lista de embeddings vacía")
                    else:
                        logger.error(f"❌ 'embeddings' no encontrado en respuesta: {data}")
                        raise Exception(f"Campo 'embeddings' no encontrado: {list(data.keys())}")
                        
                except json.JSONDecodeError as e:
                    logger.error(f"❌ Error parseando JSON: {e}")
                    logger.error(f"❌ Response text: {response_text}")
                    raise Exception(f"Error parseando respuesta JSON: {e}")
                
        except httpx.TimeoutException:
            logger.error("❌ Timeout en request a Cohere")
            raise Exception("Timeout al conectar con Cohere")
        except httpx.HTTPStatusError as e:
            logger.error(f"❌ HTTP Status Error: {e}")
            raise Exception(f"Error HTTP Cohere: {e}")
        except Exception as e:
            logger.error(f"❌ Error inesperado completo: {type(e).__name__}: {str(e)}")
            raise Exception(f"Error en Cohere: {str(e)}")
    
    async def health_check(self) -> bool:
        """Health check con debug"""
        try:
            if not self.api_key:
                logger.warning("❌ Health check: API key no configurado")
                return False
            
            logger.info("🔍 Iniciando health check...")
            embedding = await self.get_text_embedding("health check test")
            
            success = isinstance(embedding, list) and len(embedding) > 0
            logger.info(f"🔍 Health check resultado: {'✅' if success else '❌'}")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Health check falló: {type(e).__name__}: {str(e)}")
            return False
    
    async def get_api_info(self) -> Dict[str, Any]:
        """Info del servicio con debug"""
        return {
            "service": "Cohere AI",
            "model": self.model,
            "dimension": "variable (depende del modelo)",
            "api_configured": bool(self.api_key),
            "base_url": self.base_url,
            "provider": "cohere",
            "tier": "free",
            "api_key_length": len(self.api_key) if self.api_key else 0,
            "note": "1000 requests gratuitos/mes - con debug extendido"
        }