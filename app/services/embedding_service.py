# app/services/embedding_service.py
import logging
import tempfile
import os
from typing import List, Dict, Any
import asyncio
from app.core.config import settings

# Importaciones de Google Cloud (instalar en requirements.txt)
try:
    import vertexai
    from vertexai.vision_models import Image as VertexImage, MultiModalEmbeddingModel
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False
    vertexai = None
    VertexImage = None
    MultiModalEmbeddingModel = None

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Google Multimodal Embeddings - 1408 dimensiones"""
    
    def __init__(self):
        self.project_id = settings.GOOGLE_CLOUD_PROJECT_ID
        self.location = settings.GOOGLE_CLOUD_LOCATION
        self.model = None
        self.cost_per_embedding = 0.0001  # $0.0001 por embedding
        self.embedding_count = 0
        self.timeout = settings.REQUEST_TIMEOUT
        
        if not GOOGLE_AVAILABLE:
            logger.error("❌ Google Vertex AI no disponible. Instalar: pip install google-cloud-aiplatform")
            return
        
        if not self.project_id:
            logger.warning("⚠️ GOOGLE_CLOUD_PROJECT_ID no configurado")
            return
        
        # Inicializar en background para no bloquear startup
        asyncio.create_task(self._initialize_google_model())
    
    async def _initialize_google_model(self):
        """Inicializar modelo Google Multimodal de forma asíncrona"""
        try:
            logger.info("🔄 Inicializando Google Multimodal Embeddings...")
            
            # Inicializar Vertex AI
            vertexai.init(project=self.project_id, location=self.location)
            
            # Cargar modelo
            self.model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")
            
            logger.info("✅ Google Multimodal Embeddings inicializado")
            logger.info(f"📊 Proyecto: {self.project_id}")
            logger.info(f"📊 Región: {self.location}")
            logger.info(f"📊 Dimensiones: 1408")
            
        except Exception as e:
            logger.error(f"❌ Error inicializando Google Multimodal: {e}")
            self.model = None
    
    async def get_image_embedding(self, image_data: bytes) -> List[float]:
        """Generar embedding de imagen con Google Multimodal (1408 dims)"""
        
        # Esperar inicialización del modelo
        max_wait = 30
        waited = 0
        while self.model is None and waited < max_wait:
            logger.info("⏳ Esperando inicialización de Google Multimodal...")
            await asyncio.sleep(2)
            waited += 2
        
        if self.model is None:
            raise Exception("❌ Google Multimodal no disponible")
        
        try:
            # Crear archivo temporal para la imagen
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                temp_file.write(image_data)
                temp_path = temp_file.name
            
            try:
                logger.info("🔄 Generando embedding con Google Multimodal...")
                
                # Cargar imagen para Vertex AI
                vertex_image = VertexImage.load_from_file(temp_path)
                
                # Generar embedding con dimensiones máximas (1408)
                embeddings_response = self.model.get_embeddings(
                    image=vertex_image,
                    dimension=1408  # Dimensión exacta de tu BD
                )
                
                # Extraer el embedding
                embedding = embeddings_response.image_embedding
                
                # Convertir a lista
                if hasattr(embedding, 'tolist'):
                    embedding_list = embedding.tolist()
                else:
                    embedding_list = list(embedding)
                
                # Verificar dimensiones
                if len(embedding_list) != 1408:
                    logger.warning(f"⚠️ Embedding dimensión inesperada: {len(embedding_list)}")
                
                self.embedding_count += 1
                
                logger.info(f"✅ Google Multimodal embedding generado: {len(embedding_list)} dims")
                logger.info(f"💰 Costo acumulado: ${self.embedding_count * self.cost_per_embedding:.4f}")
                
                return embedding_list
                
            finally:
                # Limpiar archivo temporal
                try:
                    os.unlink(temp_path)
                except:
                    pass
                
        except Exception as e:
            logger.error(f"❌ Error generando embedding con Google Multimodal: {e}")
            raise Exception(f"Error en Google Multimodal: {str(e)}")
    
    async def get_text_embedding(self, text: str) -> List[float]:
        """Generar embedding de texto con Google Multimodal"""
        
        if self.model is None:
            raise Exception("❌ Google Multimodal no disponible")
        
        try:
            logger.info(f"🔄 Generando embedding de texto: '{text[:50]}...'")
            
            # Generar embedding de texto
            embeddings_response = self.model.get_embeddings(
                text=text,
                dimension=1408
            )
            
            # Extraer embedding
            embedding = embeddings_response.text_embedding
            
            if hasattr(embedding, 'tolist'):
                embedding_list = embedding.tolist()
            else:
                embedding_list = list(embedding)
            
            logger.info(f"✅ Texto embedding generado: {len(embedding_list)} dims")
            
            return embedding_list
            
        except Exception as e:
            logger.error(f"❌ Error en texto embedding Google Multimodal: {e}")
            raise Exception(f"Error en texto Google Multimodal: {str(e)}")
    
    async def health_check(self) -> bool:
        """Health check de Google Multimodal"""
        try:
            if not GOOGLE_AVAILABLE:
                logger.warning("❌ Google Vertex AI no disponible")
                return False
            
            if not self.project_id:
                logger.warning("❌ GOOGLE_CLOUD_PROJECT_ID no configurado")
                return False
            
            # Verificar si el modelo está inicializado
            model_ready = self.model is not None
            logger.info(f"🔍 Google Multimodal health check: {'✅' if model_ready else '❌'}")
            
            return model_ready
            
        except Exception as e:
            logger.error(f"❌ Health check Google Multimodal falló: {e}")
            return False
    
    async def get_api_info(self) -> Dict[str, Any]:
        """Información del servicio Google Multimodal"""
        return {
            "service": "Google Multimodal Embeddings",
            "model": "multimodalembedding@001",
            "dimension": 1408,
            "api_configured": self.model is not None,
            "provider": "google_vertex_ai",
            "project_id": self.project_id,
            "location": self.location,
            "cost_per_embedding": self.cost_per_embedding,
            "embeddings_generated": self.embedding_count,
            "google_vertex_available": GOOGLE_AVAILABLE,
            "note": "Compatible con BD de 1408 dimensiones"
        }