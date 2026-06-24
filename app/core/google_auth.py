# app/core/google_auth.py
import os
import json
import tempfile
import atexit
from app.core.config import settings

def setup_google_credentials():
    """Configurar credenciales de Google Cloud en Render"""
    
    # Obtener JSON de credenciales desde variable de entorno
    credentials_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    
    if credentials_json:
        try:
            # Crear archivo temporal con las credenciales
            credentials_dict = json.loads(credentials_json)
            
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
                json.dump(credentials_dict, f)
                credentials_file = f.name

            # Permisos restrictivos explicitos (solo el owner) y limpieza al salir
            os.chmod(credentials_file, 0o600)
            atexit.register(lambda: os.path.exists(credentials_file) and os.unlink(credentials_file))

            # Configurar variable de entorno para Google Cloud
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_file
            del credentials_json, credentials_dict

            print(f"✅ Credenciales Google Cloud configuradas (permisos 600)")
            return True
            
        except Exception as e:
            print(f"❌ Error configurando credenciales: {e}")
            return False
    else:
        print("⚠️ GOOGLE_APPLICATION_CREDENTIALS_JSON no configurado")
        return False