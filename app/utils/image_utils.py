# app/utils/image_utils.py
from fastapi import HTTPException, UploadFile
from PIL import Image
import io
import logging
from typing import Tuple
from app.core.config import settings
from typing import Tuple, Dict

logger = logging.getLogger(__name__)

async def validate_and_process_image(image: UploadFile) -> Tuple[bytes, Dict]:
    """
    Validar y procesar imagen subida
    
    Returns:
        Tuple de (image_data, image_info)
    """
    
    # Verificar que es un archivo de imagen
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, 
            detail="El archivo debe ser una imagen (JPEG, PNG, etc.)"
        )
    
    # Leer datos de la imagen
    image_data = await image.read()
    
    # Verificar tamaño del archivo
    if len(image_data) > settings.MAX_IMAGE_SIZE:
        max_mb = settings.MAX_IMAGE_SIZE / (1024 * 1024)
        raise HTTPException(
            status_code=400, 
            detail=f"Imagen muy grande. Tamaño máximo: {max_mb:.1f}MB"
        )
    
    # Validar que es una imagen válida y obtener dimensiones
    try:
        pil_image = Image.open(io.BytesIO(image_data))
        width, height = pil_image.size
        format_name = pil_image.format
        
        # Verificar la imagen
        pil_image.verify()
        
        image_info = {
            "filename": image.filename,
            "content_type": image.content_type,
            "size_bytes": len(image_data),
            "width": width,
            "height": height,
            "format": format_name
        }
        
        logger.info(f"📷 Imagen procesada: {image.filename} ({width}x{height}, {len(image_data)} bytes)")
        return image_data, image_info
        
    except Exception as e:
        logger.error(f"❌ Error validando imagen: {e}")
        raise HTTPException(
            status_code=400, 
            detail="Archivo de imagen inválido o corrupto"
        )

def get_confidence_level(score: float) -> str:
    """Convertir score de similitud a nivel de confianza"""
    percentage = score * 100
    
    if percentage >= 85:
        return "very_high"
    elif percentage >= 70:
        return "high"
    elif percentage >= 50:
        return "medium"
    elif percentage >= 30:
        return "low"
    else:
        return "very_low"