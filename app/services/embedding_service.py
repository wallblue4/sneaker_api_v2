# app/services/embedding_service.py
import httpx
import base64
import logging
from typing import List, Optional, Dict, Any
from PIL import Image
import io
import asyncio
from app.core.config import settings

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Servicio para generar embeddings usando HuggingFace Inference API"""
    
    def __init__(self):
        self.api_key = settings.HF_API_TOKEN
        self.timeout = settings.REQUEST_TIMEOUT
        
        # Usar Jina CLIP v2 en HuggingFace - 1024 dimensiones
        self.model_url = "https://api-inference.huggingface.co/models/jinaai/jina-clip-v2"
        self.model_name = "jinaai/jina-clip-v2"
        
        if not self.api_key:
            logger.warning("⚠️ HF_API_TOKEN no configurado")
    
    async def get_image_embedding(self, image_data: bytes) -> List[float]:
        """
        Generar embedding para imagen usando HuggingFace
        Retorna exactamente 1024 dimensiones (compatible con tu BD)
        """
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Convertir imagen a base64
            image_b64 = base64.b64encode(image_data).decode('utf-8')
            
            payload = {
                "inputs": {
                    "image": image_b64
                },
                "options": {
                    "wait_for_model": True,
                    "use_cache": True
                }
            }
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                logger.info("🔄 Generando embedding de imagen con HuggingFace...")
                
                response = await client.post(
                    self.model_url,
                    json=payload, 
                    headers=headers
                )
                
                if response.status_code == 503:
                    # Modelo cargándose, esperar
                    logger.info("⏳ Modelo cargándose, reintentando en 3s...")
                    await asyncio.sleep(3)
                    response = await client.post(self.model_url, json=payload, headers=headers)
                
                response.raise_for_status()
                
                # Procesar respuesta de HuggingFace
                result = response.json()
                
                # HuggingFace puede retornar diferentes formatos
                if isinstance(result, list):
                    embedding = result[0] if isinstance(result[0], list) else result
                elif isinstance(result, dict) and "embeddings" in result:
                    embedding = result["embeddings"][0]
                else:
                    embedding = result
                
                # Verificar dimensiones
                if len(embedding) != 1024:
                    logger.warning(f"⚠️ Embedding dimensión inesperada: {len(embedding)}")
                
                logger.info(f"✅ Embedding generado: dimensión {len(embedding)}")
                return embedding
                
        except httpx.TimeoutException:
            logger.error("❌ Timeout generando embedding")
            raise Exception("Timeout al generar embedding")
        except httpx.HTTPStatusError as e:
            logger.error(f"❌ Error HTTP {e.response.status_code}: {e.response.text}")
            raise Exception(f"Error en API HuggingFace: {e.response.status_code}")
        except Exception as e:
            logger.error(f"❌ Error inesperado generando embedding: {e}")
            raise
    
    async def get_text_embedding(self, text: str) -> List[float]:
        """Generar embedding para texto - 1024 dimensiones"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "inputs": text,
                "options": {
                    "wait_for_model": True,
                    "use_cache": True
                }
            }
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                logger.info(f"🔄 Generando embedding de texto: '{text[:50]}...'")
                
                response = await client.post(
                    self.model_url,
                    json=payload, 
                    headers=headers
                )
                
                if response.status_code == 503:
                    await asyncio.sleep(3)
                    response = await client.post(self.model_url, json=payload, headers=headers)
                
                response.raise_for_status()
                
                result = response.json()
                
                # Procesar respuesta similar a imagen
                if isinstance(result, list):
                    embedding = result[0] if isinstance(result[0], list) else result
                elif isinstance(result, dict) and "embeddings" in result:
                    embedding = result["embeddings"][0]
                else:
                    embedding = result
                
                return embedding
                
        except Exception as e:
            logger.error(f"❌ Error generando embedding de texto: {e}")
            raise
    
    async def health_check(self) -> bool:
        """Verificar que HuggingFace API está funcionando"""
        try:
            if not self.api_key:
                return False
                
            headers = {"Authorization": f"Bearer {self.api_key}"}
            
            test_timeout = httpx.Timeout(10.0)
            async with httpx.AsyncClient(timeout=test_timeout) as client:
                response = await client.post(
                    self.model_url,
                    json={
                        "inputs": "health check test",
                        "options": {"wait_for_model": False}
                    },
                    headers=headers
                )
                
                # 200 = OK, 503 = modelo disponible pero cargándose
                return response.status_code in [200, 503]
                
        except Exception as e:
            logger.error(f"❌ Health check HuggingFace falló: {e}")
            return False
    
    async def get_api_info(self) -> Dict[str, Any]:
        """Información de la API"""
        return {
            "service": "HuggingFace Inference API",
            "model": self.model_name,
            "dimension": 1024,  # Confirmar 1024 dimensiones
            "api_configured": bool(self.api_key),
            "base_url": self.model_url,
            "provider": "huggingface",
            "compatible_with_bd": True
        }