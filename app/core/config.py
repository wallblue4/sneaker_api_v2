# app/core/config.py
import os
from typing import List
from pydantic_settings import BaseSettings  

class Settings(BaseSettings):
    # Google Cloud configuración
    GOOGLE_CLOUD_PROJECT_ID: str = os.getenv("GOOGLE_CLOUD_PROJECT_ID", "")
    GOOGLE_CLOUD_LOCATION: str = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    
    # Pinecone configuración (actualizada para 1408 dims)
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "sneaker-embeddings")
    EMBEDDING_DIMENSION: int = 1408  # ✅ Actualizado a Google Multimodal

    CLASSIFICATION_SEARCH_MULTIPLIER: int = 3
    CLASSIFICATION_MAX_SEARCH: int = 100
    CLASSIFICATION_BATCH_SIZE: int = 20
    CLASSIFICATION_MAX_ITERATIONS: int = 3
    
    # Configuración de aplicación
    MAX_IMAGE_SIZE: int = 10 * 1024 * 1024  # 10MB
    MAX_TOP_K: int = 20
    REQUEST_TIMEOUT: float = 60.0  # Más tiempo para Google API
    
    # Configuración de servidor
    HOST: str = "0.0.0.0"
    PORT: int = int(os.getenv("PORT", 10000))
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "production")
    
    # CORS
    ALLOWED_ORIGINS: List[str] = ["*"]  # En producción, especificar dominios exactos

    class Config:
        env_file = ".env"
        case_sensitive = True
    
    @property
    def is_development(self) -> bool:
        return self.ENVIRONMENT == "development"
    
    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"

settings = Settings()