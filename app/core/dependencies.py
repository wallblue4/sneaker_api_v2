# app/core/dependencies.py
from fastapi import HTTPException, Depends, Header
from typing import Tuple, Optional
import logging

from app.core.config import settings

# Variables globales para servicios (se inicializan en main.py)
embedding_service = None
pinecone_service = None

logger = logging.getLogger(__name__)

async def verify_api_key(x_api_key: Optional[str] = Header(None)):
    """Autenticacion de servicio. Falla cerrado: rechaza si API_KEY no esta
    configurada o si la cabecera X-API-Key no coincide."""
    if not settings.API_KEY or x_api_key != settings.API_KEY:
        raise HTTPException(status_code=403, detail="API key invalida o ausente")

async def get_services():
    """Dependency injection para obtener servicios"""
    global embedding_service, pinecone_service
    
    if not embedding_service or not pinecone_service:
        raise HTTPException(
            status_code=503, 
            detail="Servicios no disponibles. Intenta más tarde."
        )
    
    return embedding_service, pinecone_service

async def get_embedding_service():
    """Obtener solo servicio de embeddings"""
    global embedding_service
    
    if not embedding_service:
        raise HTTPException(503, "Servicio de embeddings no disponible")
    
    return embedding_service

async def get_pinecone_service():
    """Obtener solo servicio de Pinecone"""
    global pinecone_service
    
    if not pinecone_service:
        raise HTTPException(503, "Servicio de búsqueda no disponible")
    
    return pinecone_service

def set_services(embed_svc, pinecone_svc):
    """Configurar servicios globales (llamado desde main.py)"""
    global embedding_service, pinecone_service
    embedding_service = embed_svc
    pinecone_service = pinecone_svc
    logger.info("✅ Servicios configurados en dependencies")