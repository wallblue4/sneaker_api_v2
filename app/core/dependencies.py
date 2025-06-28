# app/core/dependencies.py
from fastapi import HTTPException, Depends
from typing import Tuple, Optional
import logging

# Variables globales para servicios (se inicializan en main.py)
embedding_service = None
pinecone_service = None

logger = logging.getLogger(__name__)

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