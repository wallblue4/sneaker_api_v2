# app/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import logging
import asyncio
import time

from app.core.config import settings
from app.core.dependencies import set_services
from app.api.routes import classification, health

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

# Suprimir logs excesivos de httpx
logging.getLogger("httpx").setLevel(logging.WARNING)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle management - Ultra-optimizado para Render free tier
    """
    startup_start = time.time()
    logger.info("🚀 Iniciando Sneaker Classification API v2.0")
    logger.info(f"🌍 Entorno: {settings.ENVIRONMENT}")
    
    try:
        # Importar servicios dinámicamente para ahorrar memoria
        from app.core.google_auth import setup_google_credentials
        setup_google_credentials()
        
        from app.services.embedding_service import EmbeddingService
        from app.services.pinecone_service import PineconeService
        
        logger.info("📦 Inicializando servicios con Google Multimodal...")
        
        # Inicializar servicios
        embedding_service = EmbeddingService()
        pinecone_service = PineconeService()
        
        # Configurar en dependencies
        set_services(embedding_service, pinecone_service)
        
        # Health checks en paralelo (timeout corto para no bloquear startup)
        logger.info("🔍 Verificando conectividad de servicios...")
        
        try:
            # Timeout corto para no bloquear el startup en Render
            health_tasks = [
                asyncio.wait_for(embedding_service.health_check(), timeout=10.0),
                asyncio.wait_for(pinecone_service.health_check(), timeout=10.0)
            ]
            
            health_results = await asyncio.gather(*health_tasks, return_exceptions=True)
            
            embedding_ok = health_results[0] if not isinstance(health_results[0], Exception) else False
            pinecone_ok = health_results[1] if not isinstance(health_results[1], Exception) else False
            
            if embedding_ok:
                logger.info("✅ Jina AI - Conectado")
            else:
                logger.warning("⚠️ Jina AI - No disponible")
                
            if pinecone_ok:
                stats = await pinecone_service.get_stats()
                total_vectors = stats.get('total_vectors', 'N/A')
                logger.info(f"✅ Pinecone - Conectado ({total_vectors} vectores)")
            else:
                logger.warning("⚠️ Pinecone - No disponible")
            
            # Mostrar estadísticas de startup
            startup_time = (time.time() - startup_start) * 1000
            logger.info(f"🎉 Startup completado en {startup_time:.2f}ms")
            
            # Servicio listo para recibir requests
            logger.info("🟢 API lista para recibir requests")
            
        except asyncio.TimeoutError:
            logger.warning("⏰ Timeout en health checks - continuando sin verificación")
        except Exception as e:
            logger.warning(f"⚠️ Error en health checks: {e} - continuando")
        
    except Exception as e:
        logger.error(f"❌ Error crítico durante startup: {e}")
        # No fallar completamente - permitir que la app arranque
        # Render puede reiniciar automáticamente si es necesario
        
    yield
    
    # Cleanup
    logger.info("🔄 Cerrando Sneaker Classification API v2.0")

# Crear aplicación FastAPI
app = FastAPI(
    title="Sneaker Classification API",
    description="""
    🔥 **Sneaker Classification API v2.0** - Ultra-light microservice
    
    Clasificación de zapatillas usando:
    - 🧠 **Jina AI** para embeddings multimodales
    - 🚀 **Pinecone** para búsqueda vectorial
    - ⚡ **Ultra-optimizado** para Render free tier
    
    ## Funcionalidades
    
    - 📷 **Clasificación por imagen**: Sube una foto y encuentra sneakers similares
    - 🔍 **Búsqueda por texto**: Describe el sneaker que buscas
    - 🏷️ **Filtros avanzados**: Por marca, precio, color, etc.
    - 📊 **Stats en tiempo real**: Estadísticas de la base de datos
    
    ## Ejemplos de uso
    
    - Buscar por imagen: `POST /api/v2/classify`
    - Buscar por texto: `POST /api/v2/search-text`
    - Health check: `GET /health`
    """,
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.is_development else None,
    redoc_url="/redoc" if settings.is_development else None,
    openapi_url="/openapi.json" if settings.is_development else None
)

# Middleware de seguridad
if settings.is_production:
    app.add_middleware(
        TrustedHostMiddleware, 
        allowed_hosts=["*"]  # En producción, especificar hosts exactos
    )

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Incluir routers
app.include_router(
    health.router, 
    prefix="/health", 
    tags=["🏥 Health Check"]
)

app.include_router(
    classification.router, 
    prefix="/api/v2", 
    tags=["🔍 Classification & Search"]
)

@app.get("/")
async def root():
    """
    🏠 Endpoint raíz - Información del servicio
    """
    return {
        "service": "Sneaker Classification API",
        "version": "2.0.0",
        "status": "🟢 running",
        "architecture": "serverless-optimized",
        "features": [
            "jina-ai-embeddings",
            "pinecone-vector-search", 
            "ultra-light-memory",
            "render-optimized"
        ],
        "endpoints": {
            "classify_image": "/api/v2/classify",
            "search_text": "/api/v2/search-text",
            "health": "/health",
            "docs": "/docs" if settings.is_development else "disabled_in_production"
        },
        "limits": {
            "max_image_size_mb": settings.MAX_IMAGE_SIZE / (1024 * 1024),
            "max_results": settings.MAX_TOP_K,
            "timeout_seconds": settings.REQUEST_TIMEOUT
        }
    }

@app.get("/favicon.ico")
async def favicon():
    """Favicon para evitar 404s"""
    return {"message": "👟"}

# Para development local
if __name__ == "__main__":
    import uvicorn
    
    logger.info("🔧 Iniciando en modo desarrollo...")
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.is_development,
        log_level="info",
        access_log=True
    )