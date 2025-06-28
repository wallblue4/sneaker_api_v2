FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=10000

RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app

COPY requirements.txt .

# QUITAMOS --no-deps que estaba causando problemas
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:$PORT/health/live', timeout=5)"

EXPOSE $PORT

# Comando estable sin uvloop ni httptools
CMD python -m uvicorn app.main:app \
    --host 0.0.0.0 \
    --port $PORT \
    --workers 1 \
    --loop asyncio \
    --http h11
