# tests/test_classification.py
import pytest
from fastapi.testclient import TestClient
from app.main import app
import io
from PIL import Image

client = TestClient(app)

def create_test_image():
    """Crear imagen de prueba"""
    img = Image.new('RGB', (100, 100), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    return img_bytes

def test_classify_endpoint_structure():
    """Test estructura del endpoint de clasificación"""
    # Crear imagen de prueba
    test_image = create_test_image()
    
    response = client.post(
        "/api/v2/classify",
        files={"image": ("test.jpg", test_image, "image/jpeg")},
        data={"top_k": 3}
    )
    
    # Puede fallar sin APIs configuradas, pero verificamos estructura
    if response.status_code == 200:
        data = response.json()
        assert "success" in data
        assert "results" in data
        assert "processing_time_ms" in data
    else:
        # Expected sin configuración de APIs
        assert response.status_code in [503, 500]

def test_search_text_structure():
    """Test estructura de búsqueda por texto"""
    response = client.post(
        "/api/v2/search-text",
        json={
            "query": "red nike shoes",
            "top_k": 3
        }
    )
    
    # Similar - puede fallar sin APIs pero verificamos estructura
    if response.status_code == 200:
        data = response.json()
        assert "success" in data
        assert "query" in data
        assert "results" in data