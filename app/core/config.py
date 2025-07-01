# app/core/config.py
from pydantic_settings import BaseSettings
from typing import List
import os

class Settings(BaseSettings):
    # Servidor
    HOST: str = "0.0.0.0"
    PORT: int = int(os.getenv("PORT", "8000"))
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    
    # APIs externas
##HF_API_TOKEN: str = os.getenv("HF_API_TOKEN", "")
    PINECONE_API_KEY: str = ""
    PINECONE_INDEX_NAME: str = "sneaker-embeddings"
    
    # Configuración CLIP (matching tu migración)
    EMBEDDING_DIMENSION: int = 1024  # ViT-L/14 como en tu script
    
    # Límites para Render free tier
    MAX_IMAGE_SIZE: int = 5 * 1024 * 1024  # 5MB
    REQUEST_TIMEOUT: int = 30
    DEFAULT_TOP_K: int = 5
    MAX_TOP_K: int = 20
    
    # CORS
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "*"  # En producción, especificar dominios exactos
    ]
    
    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"
    
    @property
    def is_development(self) -> bool:
        return self.ENVIRONMENT == "development"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()