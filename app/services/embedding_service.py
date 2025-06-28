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
    """Servicio para generar embeddings usando Jina AI API"""
    
    def __init__(self):
        self.api_key = settings.JINA_API_KEY
        self.base_url = "https://api.jina.ai/v1/embeddings"
        self.model = "jina-clip-v2"  # Multimodal model compatible
        self.timeout = settings.REQUEST_TIMEOUT
        
        if not self.api_key:
            logger.warning("⚠️ JINA_API_KEY no configurada")
    
    async def get_image_embedding(self, image_data: bytes) -> List[float]:
        """
        Generar embedding para imagen usando Jina AI
        
        Args:
            image_data: Bytes de la imagen
            
        Returns:
            Lista de floats representando el embedding
        """
        try:
            # Convertir imagen a base64
            image_b64 = base64.b64encode(image_data).decode('utf-8')
            
            payload = {
                "model": self.model,
                "input": [{"image": image_b64}],
                "normalized": True,
                "embedding_type": "float"
            }
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                logger.info("🔄 Generando embedding de imagen con Jina AI...")
                
                response = await client.post(
                    self.base_url, 
                    json=payload, 
                    headers=headers
                )
                response.raise_for_status()
                
                data = response.json()
                embedding = data["data"][0]["embedding"]
                
                logger.info(f"✅ Embedding generado: dimensión {len(embedding)}")
                return embedding
                
        except httpx.TimeoutException:
            logger.error("❌ Timeout generando embedding")
            raise Exception("Timeout al generar embedding")
        except httpx.HTTPStatusError as e:
            logger.error(f"❌ Error HTTP {e.response.status_code}: {e.response.text}")
            raise Exception(f"Error en API Jina: {e.response.status_code}")
        except Exception as e:
            logger.error(f"❌ Error inesperado generando embedding: {e}")
            raise
    
    async def get_text_embedding(self, text: str) -> List[float]:
        """Generar embedding para texto usando Jina AI"""
        try:
            payload = {
                "model": self.model,
                "input": [text],
                "normalized": True,
                "embedding_type": "float"
            }
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    self.base_url, 
                    json=payload, 
                    headers=headers
                )
                response.raise_for_status()
                
                data = response.json()
                return data["data"][0]["embedding"]
                
        except Exception as e:
            logger.error(f"❌ Error generando embedding de texto: {e}")
            raise
    
    async def health_check(self) -> bool:
        """Verificar que la API de Jina está funcionando"""
        try:
            if not self.api_key:
                return False
                
            # Test con texto simple y corto timeout
            test_timeout = httpx.Timeout(10.0)
            async with httpx.AsyncClient(timeout=test_timeout) as client:
                response = await client.post(
                    self.base_url,
                    json={
                        "model": self.model,
                        "input": ["test"],
                        "normalized": True
                    },
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    }
                )
                response.raise_for_status()
                return True
                
        except Exception as e:
            logger.error(f"❌ Health check Jina AI falló: {e}")
            return False
    
    async def get_api_info(self) -> Dict[str, Any]:
        """Obtener información sobre la API"""
        return {
            "service": "Jina AI Embeddings",
            "model": self.model,
            "dimension": settings.EMBEDDING_DIMENSION,
            "api_configured": bool(self.api_key),
            "base_url": self.base_url
        }