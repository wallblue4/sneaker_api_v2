# app/services/embedding_service.py
import httpx
import logging
from typing import List, Optional, Dict, Any
import asyncio
from PIL import Image
import io
import random
from app.core.config import settings

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Servicio usando HuggingFace API pública (sin token)"""
    
    def __init__(self):
        self.timeout = settings.REQUEST_TIMEOUT
        
        # Modelos públicos disponibles sin token
        self.models = [
            {
                "url": "https://api-inference.huggingface.co/models/openai/clip-vit-large-patch14",
                "name": "CLIP ViT-L/14",
                "dims": 768
            },
            {
                "url": "https://api-inference.huggingface.co/models/sentence-transformers/clip-ViT-L-14", 
                "name": "Sentence-CLIP ViT-L/14",
                "dims": 768
            },
            {
                "url": "https://api-inference.huggingface.co/models/openai/clip-vit-base-patch32",
                "name": "CLIP ViT-B/32", 
                "dims": 512
            }
        ]
        
        # Empezar con el primer modelo
        self.current_model = self.models[0]
        
        logger.info(f"🔧 Usando HuggingFace público: {self.current_model['name']}")
    
    def intelligent_expand_to_1024(self, embedding: List[float]) -> List[float]:
        """Expandir embedding a 1024 dimensiones de forma inteligente"""
        current_size = len(embedding)
        target_size = 1024
        
        if current_size == target_size:
            return embedding
        
        if current_size == 768:
            # Expandir 768 → 1024 (añadir 256)
            # Método: tomar fragmentos representativos y aplicar factor de escala
            expansion_size = target_size - current_size  # 256
            
            # Dividir el embedding en segmentos y usar algunos para expansión
            segment_size = current_size // 4  # 192 elementos por segmento
            segments = [
                embedding[i:i+segment_size] 
                for i in range(0, current_size, segment_size)
            ]
            
            # Crear expansión basada en los segmentos con factor de reducción
            expansion = []
            factor = 0.3  # Factor de reducción para la expansión
            
            for i in range(expansion_size):
                segment_idx = i % len(segments)
                element_idx = i % len(segments[segment_idx])
                expansion.append(segments[segment_idx][element_idx] * factor)
            
            result = embedding + expansion
            logger.info(f"📏 Expandido inteligentemente de {current_size} a {len(result)} dimensiones")
            return result
            
        elif current_size == 512:
            # Expandir 512 → 1024 (duplicar + variación)
            # Método: duplicar con ligera variación
            base_expansion = embedding.copy()  # Primera copia
            
            # Añadir variación sutil a la segunda mitad
            varied_expansion = [x * 0.8 + random.uniform(-0.01, 0.01) for x in embedding]
            
            result = embedding + base_expansion + varied_expansion[:512-len(embedding)]
            logger.info(f"📏 Expandido de {current_size} a {len(result)} dimensiones")
            return result[:target_size]  # Asegurar exactamente 1024
        
        else:
            # Para otros tamaños, usar padding inteligente
            if current_size < target_size:
                padding_size = target_size - current_size
                # Usar estadísticas del embedding para el padding
                mean_val = sum(embedding) / len(embedding)
                std_val = (sum((x - mean_val) ** 2 for x in embedding) / len(embedding)) ** 0.5
                
                # Padding con distribución similar
                padding = [
                    mean_val + random.uniform(-std_val/10, std_val/10) 
                    for _ in range(padding_size)
                ]
                
                result = embedding + padding
                logger.info(f"📏 Expandido con padding estadístico de {current_size} a {len(result)}")
                return result
            else:
                # Truncar si es muy grande
                result = embedding[:target_size]
                logger.info(f"📏 Truncado de {current_size} a {len(result)} dimensiones")
                return result
    
    async def get_image_embedding(self, image_data: bytes) -> List[float]:
        """Generar embedding de imagen con fallback automático"""
        
        for attempt, model in enumerate(self.models):
            try:
                logger.info(f"🔄 Intentando {model['name']} (intento {attempt + 1})...")
                
                headers = {
                    "Content-Type": "application/octet-stream",
                    "User-Agent": "Mozilla/5.0 (compatible; SneakerAPI/2.0)"
                }
                
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        model['url'],
                        content=image_data,
                        headers=headers
                    )
                    
                    # Manejar diferentes códigos de estado
                    if response.status_code == 503:
                        wait_time = 10 + (attempt * 5)  # Espera incremental
                        logger.info(f"⏳ Modelo cargándose, esperando {wait_time}s...")
                        await asyncio.sleep(wait_time)
                        
                        response = await client.post(
                            model['url'],
                            content=image_data,
                            headers=headers
                        )
                    
                    elif response.status_code == 429:
                        wait_time = 30 + (attempt * 10)
                        logger.warning(f"⚠️ Rate limit, esperando {wait_time}s...")
                        await asyncio.sleep(wait_time)
                        continue  # Probar siguiente modelo
                    
                    if response.status_code == 200:
                        embedding = response.json()
                        
                        if isinstance(embedding, list) and len(embedding) > 0:
                            # Expandir a 1024 dimensiones
                            final_embedding = self.intelligent_expand_to_1024(embedding)
                            
                            # Actualizar modelo actual exitoso
                            self.current_model = model
                            
                            logger.info(f"✅ {model['name']} exitoso - dimensión final: {len(final_embedding)}")
                            return final_embedding
                        else:
                            logger.warning(f"⚠️ {model['name']} retornó formato inesperado: {type(embedding)}")
                            continue
                    else:
                        logger.warning(f"⚠️ {model['name']} falló con status {response.status_code}")
                        continue
                        
            except Exception as e:
                logger.warning(f"⚠️ Error con {model['name']}: {e}")
                continue
        
        # Si todos los modelos fallan
        raise Exception("❌ Todos los modelos de HuggingFace fallaron. Servicio temporalmente no disponible.")
    
    async def get_text_embedding(self, text: str) -> List[float]:
        """Generar embedding de texto con fallback"""
        
        for model in self.models:
            try:
                headers = {
                    "Content-Type": "application/json",
                    "User-Agent": "Mozilla/5.0 (compatible; SneakerAPI/2.0)"
                }
                
                payload = {"inputs": text}
                
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        model['url'],
                        json=payload,
                        headers=headers
                    )
                    
                    if response.status_code == 503:
                        await asyncio.sleep(10)
                        response = await client.post(model['url'], json=payload, headers=headers)
                    
                    if response.status_code == 200:
                        embedding = response.json()
                        
                        if isinstance(embedding, list):
                            final_embedding = self.intelligent_expand_to_1024(embedding)
                            logger.info(f"✅ Texto embedding generado: {len(final_embedding)} dims")
                            return final_embedding
                    
            except Exception as e:
                logger.warning(f"⚠️ Error texto con {model['name']}: {e}")
                continue
        
        raise Exception("❌ No se pudo generar embedding de texto")
    
    async def health_check(self) -> bool:
        """Health check robusto sin token"""
        try:
            # Crear imagen de prueba muy pequeña
            test_img = Image.new('RGB', (64, 64), color=(128, 128, 128))
            buffer = io.BytesIO()
            test_img.save(buffer, format='JPEG', quality=50)
            test_data = buffer.getvalue()
            
            # Probar solo el primer modelo para health check
            headers = {
                "Content-Type": "application/octet-stream",
                "User-Agent": "Mozilla/5.0 (compatible; SneakerAPI/2.0)"
            }
            
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(
                    self.models[0]['url'],
                    content=test_data,
                    headers=headers
                )
                
                # 200 = OK, 503 = disponible pero cargándose
                success = response.status_code in [200, 503]
                
                if success:
                    logger.info(f"✅ Health check exitoso: status {response.status_code}")
                else:
                    logger.warning(f"⚠️ Health check falló: status {response.status_code}")
                
                return success
                
        except Exception as e:
            logger.error(f"❌ Health check error: {e}")
            return False
    
    async def get_api_info(self) -> Dict[str, Any]:
        """Información del servicio"""
        return {
            "service": "HuggingFace Inference API (Public)",
            "model": self.current_model['name'],
            "dimension": 1024,
            "api_configured": True,
            "base_url": self.current_model['url'],
            "provider": "huggingface_public",
            "fallback_models": len(self.models),
            "note": "Usando API pública con expansión inteligente a 1024 dims"
        }