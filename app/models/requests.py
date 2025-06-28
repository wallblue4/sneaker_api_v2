# app/models/requests.py
from pydantic import BaseModel, Field
from typing import Optional

class SearchByTextRequest(BaseModel):
    """Request para búsqueda por texto"""
    query: str = Field(..., min_length=1, max_length=200, description="Descripción del sneaker a buscar")
    top_k: int = Field(5, ge=1, le=20, description="Número de resultados")
    brand: Optional[str] = Field(None, description="Filtrar por marca")
    min_price: Optional[float] = Field(None, ge=0, description="Precio mínimo")
    max_price: Optional[float] = Field(None, ge=0, description="Precio máximo")

class SearchByBrandRequest(BaseModel):
    """Request para filtrar por marca"""
    brand: str = Field(..., min_length=1, description="Nombre de la marca")
    top_k: int = Field(5, ge=1, le=20)