# app/models/responses.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class ConfidenceLevel(str, Enum):
    VERY_HIGH = "very_high"
    HIGH = "high" 
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"

class SneakerResult(BaseModel):
    """Resultado individual de clasificación"""
    rank: int = Field(..., description="Posición en el ranking")
    similarity_score: float = Field(..., ge=0, le=1, description="Score de similitud (0-1)")
    confidence_percentage: float = Field(..., ge=0, le=100, description="Porcentaje de confianza")
    confidence_level: ConfidenceLevel = Field(..., description="Nivel de confianza")
    
    # Información del sneaker
    model_name: str = Field(..., description="Nombre del modelo")
    brand: str = Field(..., description="Marca")
    color: Optional[str] = Field(None, description="Color")
    size: Optional[str] = Field(None, description="Talla")
    price: Optional[float] = Field(None, ge=0, description="Precio")
    description: Optional[str] = Field(None, description="Descripción")
    image_path: Optional[str] = Field(None, description="Ruta de la imagen")
    
    # Metadata adicional
    original_db_id: Optional[int] = Field(None, description="ID en base de datos original")

class ImageInfo(BaseModel):
    """Información de la imagen procesada"""
    filename: Optional[str] = None
    content_type: Optional[str] = None
    size_bytes: int = Field(..., ge=0)
    width: Optional[int] = Field(None, ge=1)
    height: Optional[int] = Field(None, ge=1)
    format: Optional[str] = None

class ClassificationResponse(BaseModel):
    """Respuesta completa de clasificación"""
    success: bool = True
    processing_time_ms: float = Field(..., ge=0, description="Tiempo de procesamiento en ms")
    timestamp: datetime = Field(default_factory=datetime.now)
    
    results: List[SneakerResult] = Field(..., description="Resultados de clasificación")
    total_matches_found: int = Field(..., ge=0, description="Total de coincidencias")
    
    query_info: ImageInfo = Field(..., description="Información de la imagen consultada")
    
    # Información del modelo
    model_info: Dict[str, Any] = Field(default_factory=dict)

class SearchResponse(BaseModel):
    """Respuesta de búsqueda por texto"""
    success: bool = True
    processing_time_ms: float = Field(..., ge=0)
    timestamp: datetime = Field(default_factory=datetime.now)
    
    query: str = Field(..., description="Texto de búsqueda")
    results: List[SneakerResult] = Field(...)
    total_matches_found: int = Field(..., ge=0)
    
    filters_applied: Dict[str, Any] = Field(default_factory=dict)

class HealthResponse(BaseModel):
    """Respuesta de health check"""
    status: str = Field(..., description="Estado general del servicio")
    timestamp: datetime = Field(default_factory=datetime.now)
    
    services: Dict[str, bool] = Field(..., description="Estado de servicios externos")
    service_info: Dict[str, Any] = Field(default_factory=dict)
    
    stats: Dict[str, Any] = Field(default_factory=dict)
    version: str = "2.0.0"

class ErrorResponse(BaseModel):
    """Respuesta de error"""
    success: bool = False
    error: str = Field(..., description="Mensaje de error")
    detail: Optional[str] = Field(None, description="Detalle adicional del error")
    timestamp: datetime = Field(default_factory=datetime.now)
    error_code: Optional[str] = Field(None, description="Código de error")