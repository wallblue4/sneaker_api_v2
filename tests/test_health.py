# tests/test_health.py
import pytest
import asyncio
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root_endpoint():
    """Test endpoint raíz"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "Sneaker Classification API"
    assert data["version"] == "2.0.0"

def test_health_live():
    """Test liveness probe"""
    response = client.get("/health/live")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "alive"

def test_health_ready():
    """Test readiness probe"""
    response = client.get("/health/ready")
    # Puede fallar si los servicios externos no están configurados
    assert response.status_code in [200, 503]