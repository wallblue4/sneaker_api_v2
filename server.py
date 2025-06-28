import os
import sys

def main():
    """Punto de entrada para Render"""
    
    os.environ.setdefault("ENVIRONMENT", "production")
    port = int(os.environ.get("PORT", 10000))
    
    print(f"🚀 Iniciando Sneaker Classification API en puerto {port}")
    
    try:
        from app.main import app
        print("✅ App importada correctamente")
    except Exception as e:
        print(f"❌ Error importando app: {e}")
        sys.exit(1)
    
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        workers=1,
        loop="asyncio",
        log_level="info",
        access_log=False
    )

if __name__ == "__main__":
    main()
