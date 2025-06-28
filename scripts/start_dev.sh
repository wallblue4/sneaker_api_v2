# scripts/start_dev.sh
#!/bin/bash

echo "🚀 Iniciando Sneaker Classification API v2.0 en desarrollo"
echo "=" * 60

# Activar virtual environment
source venv/bin/activate

# Verificar que las dependencias están instaladas
echo "📦 Verificando dependencias..."
pip list | grep -E "(fastapi|uvicorn|pinecone|httpx)" || {
    echo "❌ Dependencias faltantes. Instalando..."
    pip install -r requirements.txt
}

# Verificar archivo .env
if [ ! -f ".env" ]; then
    echo "❌ Archivo .env no encontrado. Copiando ejemplo..."
    cp .env.example .env
    echo "⚠️  EDITA .env con tus API keys antes de continuar"
    exit 1
fi

# Verificar API keys
if grep -q "your_.*_api_key_here" .env; then
    echo "❌ API keys no configuradas en .env"
    echo "⚠️  Configura JINA_API_KEY y PINECONE_API_KEY"
    exit 1
fi

echo "✅ Configuración verificada"
echo ""
echo "🌐 Iniciando servidor en http://localhost:8000"
echo "📚 Documentación en http://localhost:8000/docs"
echo "🏥 Health check en http://localhost:8000/health"
echo ""
echo "Para parar el servidor: Ctrl+C"
echo ""

# Iniciar servidor
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000