# app/api/routes/health.py - Versión corregida
from fastapi import APIRouter, Depends
from typing import Dict, Any
import asyncio
import time
import logging

from app.core.dependencies import get_services
from app.models.responses import HealthResponse
from app.core.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/")
async def health_check():
    """Health check completo del servicio - versión simplificada"""
    start_time = time.time()
    
    try:
        embedding_service, pinecone_service = await get_services()
        
        logger.info("🔍 Verificando health de servicios externos...")
        
        # Health checks con mejor manejo de errores
        try:
            embedding_ok = await asyncio.wait_for(embedding_service.health_check(), timeout=5.0)
        except:
            embedding_ok = False
            
        try:
            pinecone_ok = await asyncio.wait_for(pinecone_service.health_check(), timeout=5.0)
        except:
            pinecone_ok = False
        
        # Información básica de servicios - sin objetos complejos
        service_info = {}
        stats_info = {}
        
        if embedding_ok:
            try:
                api_info = await embedding_service.get_api_info()
                # Solo datos serializables
                service_info["jina_ai"] = {
                    "service": api_info.get("service", "Jina AI"),
                    "model": api_info.get("model", "jina-clip-v2"),
                    "dimension": api_info.get("dimension", 1024),
                    "api_configured": api_info.get("api_configured", False)
                }
            except:
                service_info["jina_ai"] = {"status": "connected_but_info_unavailable"}
                
        if pinecone_ok:
            try:
                pinecone_info = await pinecone_service.get_service_info()
                pinecone_stats = await pinecone_service.get_stats()
                
                # Solo datos básicos y serializables
                service_info["pinecone"] = {
                    "service": "Pinecone Vector Search",
                    "index_name": pinecone_info.get("index_name", settings.PINECONE_INDEX_NAME),
                    "api_configured": pinecone_info.get("api_configured", False)
                }
                
                # Stats limpios
                stats_info = {
                    "total_vectors": pinecone_stats.get("total_vectors", 0),
                    "dimension": pinecone_stats.get("dimension", 0),
                    "index_fullness": pinecone_stats.get("index_fullness", 0.0)
                }
            except Exception as e:
                logger.warning(f"Error getting pinecone info: {e}")
                service_info["pinecone"] = {"status": "connected_but_info_unavailable"}
                stats_info = {"error": "stats_unavailable"}
        
        # Determinar estado general
        overall_status = "healthy"
        if not embedding_ok or not pinecone_ok:
            overall_status = "degraded"
        if not embedding_ok and not pinecone_ok:
            overall_status = "unhealthy"
        
        processing_time = (time.time() - start_time) * 1000
        
        # Respuesta completamente serializable
        return {
            "status": overall_status,
            "timestamp": time.time(),
            "services": {
                "jina_ai": embedding_ok,
                "pinecone": pinecone_ok
            },
            "service_info": service_info,
            "stats": {
                "pinecone": stats_info,
                "health_check_time_ms": round(processing_time, 2),
                "config": {
                    "max_image_size_mb": settings.MAX_IMAGE_SIZE / (1024 * 1024),
                    "max_top_k": settings.MAX_TOP_K,
                    "embedding_dimension": settings.EMBEDDING_DIMENSION
                }
            },
            "version": "2.0.0"
        }
        
    except Exception as e:
        logger.error(f"❌ Error en health check: {e}")
        return {
            "status": "error",
            "timestamp": time.time(),
            "services": {"jina_ai": False, "pinecone": False},
            "service_info": {"error": str(e)},
            "stats": {},
            "version": "2.0.0"
        }

@router.get("/live")
async def liveness_probe():
    """Liveness probe simple"""
    return {
        "status": "alive", 
        "timestamp": time.time(),
        "version": "2.0.0",
        "environment": settings.ENVIRONMENT
    }

@router.get("/ready")
async def readiness_probe():
    """Readiness probe"""
    try:
        embedding_service, pinecone_service = await get_services()
        return {"status": "ready", "timestamp": time.time()}
    except:
        return {"status": "not_ready", "timestamp": time.time()}, 503