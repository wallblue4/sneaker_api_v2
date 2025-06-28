# scripts/test_local.py
import asyncio
import httpx
import time
import json

async def test_api():
    """Test completo de la API local"""
    base_url = "http://localhost:8000"
    
    print("🧪 Testing Sneaker Classification API v2.0")
    print("=" * 50)
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        
        # 1. Test root endpoint
        print("\n1️⃣ Testing root endpoint...")
        try:
            response = await client.get(f"{base_url}/")
            print(f"✅ Root: {response.status_code}")
            print(json.dumps(response.json(), indent=2))
        except Exception as e:
            print(f"❌ Root failed: {e}")
        
        # 2. Test health check
        print("\n2️⃣ Testing health check...")
        try:
            response = await client.get(f"{base_url}/health/")
            print(f"✅ Health: {response.status_code}")
            data = response.json()
            print(f"Status: {data['status']}")
            print(f"Services: {data['services']}")
        except Exception as e:
            print(f"❌ Health failed: {e}")
        
        # 3. Test text search
        print("\n3️⃣ Testing text search...")
        try:
            response = await client.post(
                f"{base_url}/api/v2/search-text",
                json={
                    "query": "red nike air max",
                    "top_k": 3
                }
            )
            print(f"✅ Text search: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"Found {data['total_matches_found']} results")
                for result in data['results'][:2]:  # Mostrar solo 2 primeros
                   print(f"  - {result['model_name']} ({result['brand']}) - {result['confidence_percentage']:.1f}%")
        except Exception as e:
           print(f"❌ Text search failed: {e}")
       
       # 4. Test stats
        print("\n4️⃣ Testing database stats...")
        try:
            response = await client.get(f"{base_url}/api/v2/stats")
            print(f"✅ Stats: {response.status_code}")
            if response.status_code == 200:
               data = response.json()
               stats = data['database_stats']
               print(f"Total vectors: {stats.get('total_vectors', 'N/A')}")
               print(f"Dimension: {stats.get('dimension', 'N/A')}")
        except Exception as e:
           print(f"❌ Stats failed: {e}")
       
       # 5. Test image classification (requiere imagen)
        print("\n5️⃣ Testing image classification...")
        try:
           # Crear imagen de prueba
           from PIL import Image
           import io
           
           # Crear imagen simple de prueba
           img = Image.new('RGB', (200, 200), color='red')
           img_bytes = io.BytesIO()
           img.save(img_bytes, format='JPEG')
           img_bytes.seek(0)
           
           files = {"image": ("test_sneaker.jpg", img_bytes, "image/jpeg")}
           data = {"top_k": 3}
           
           response = await client.post(
               f"{base_url}/api/v2/classify",
               files=files,
               data=data
           )
           print(f"✅ Image classification: {response.status_code}")
           if response.status_code == 200:
               result = response.json()
               print(f"Processing time: {result['processing_time_ms']:.2f}ms")
               print(f"Found {result['total_matches_found']} matches")
               for match in result['results'][:2]:
                   print(f"  - {match['model_name']} ({match['brand']}) - {match['confidence_percentage']:.1f}%")
        except Exception as e:
           print(f"❌ Image classification failed: {e}")

if __name__ == "__main__":
   asyncio.run(test_api())