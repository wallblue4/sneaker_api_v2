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
        
        # Usar CLIP que SÍ está disponible en HuggingFace
        self.model_url = "https://api-inference.huggingface.co/models/openai/clip-vit-large-patch14"
        self.model_name = "openai/clip-vit-large-patch14"
        
        if not self.api_key:
            logger.warning("⚠️ HF_API_TOKEN no configurado")
    
    async def get_image_embedding(self, image_data: bytes) -> List[float]:
        """
        Generar embedding usando CLIP en HuggingFace
        """
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/octet-stream"  # Para imágenes binarias
            }
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                logger.info("🔄 Generando embedding de imagen con HuggingFace CLIP...")
                
                # Enviar imagen directamente como bytes
                response = await client.post(
                    self.model_url,
                    content=image_data,  # Enviar bytes directamente
                    headers=headers
                )
                
                if response.status_code == 503:
                    logger.info("⏳ Modelo cargándose, reintentando en 5s...")
                    await asyncio.sleep(5)
                    response = await client.post(self.model_url, content=image_data, headers=headers)
                
                response.raise_for_status()
                
                # CLIP retorna embedding directo
                embedding = response.json()
                
                # Verificar que sea lista
                if not isinstance(embedding, list):
                    raise Exception(f"Formato inesperado: {type(embedding)}")
                
                # CLIP ViT-L/14 produce 768, necesitamos expandir a 1024
                if len(embedding) == 768:
                    # Expandir a 1024 con padding cero
                    embedding = embedding + [0.0] * (1024 - 768)
                    logger.info("📏 Expandido de 768 a 1024 dimensiones")
                
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
        """Generar embedding para texto"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Para texto, usar formato JSON
            payload = {
                "inputs": text
            }
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                logger.info(f"🔄 Generando embedding de texto: '{text[:50]}...'")
                
                response = await client.post(
                    self.model_url,
                    json=payload,
                    headers=headers
                )
                
                if response.status_code == 503:
                    await asyncio.sleep(5)
                    response = await client.post(self.model_url, json=payload, headers=headers)
                
                response.raise_for_status()
                
                embedding = response.json()
                
                # Expandir si es necesario
                if len(embedding) == 768:
                    embedding = embedding + [0.0] * (1024 - 768)
                
                return embedding
                
        except Exception as e:
            logger.error(f"❌ Error generando embedding de texto: {e}")
            raise
    
    async def health_check(self) -> bool:
        """Health check simplificado"""
        try:
            if not self.api_key:
                return False
                
            headers = {"Authorization": f"Bearer {self.api_key}"}
            
            # Test con imagen pequeña
            test_image = Image.new('RGB', (100, 100), color='red')
            img_buffer = io.BytesIO()
            test_image.save(img_buffer, format='JPEG')
            test_data = img_buffer.getvalue()
            
            test_timeout = httpx.Timeout(15.0)
            async with httpx.AsyncClient(timeout=test_timeout) as client:
                response = await client.post(
                    self.model_url,
                    content=test_data,
                    headers={**headers, "Content-Type": "application/octet-stream"}
                )
                
                return response.status_code in [200, 503]
                
        except Exception as e:
            logger.error(f"❌ Health check HuggingFace falló: {e}")
            return False
    
    async def get_api_info(self) -> Dict[str, Any]:
        """Información de la API"""
        return {
            "service": "HuggingFace Inference API",
            "model": self.model_name,
            "dimension": 1024,  # Expandido desde 768
            "api_configured": bool(self.api_key),
            "base_url": self.model_url,
            "provider": "huggingface",
            "note": "CLIP ViT-L/14 expandido de 768 a 1024 dims"
        }