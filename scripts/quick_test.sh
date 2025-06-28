# scripts/quick_test.sh
#!/bin/bash

echo "🧪 Quick API Test"

# Test básico con curl
echo "1️⃣ Testing root endpoint..."
curl -s http://localhost:8000/ | jq '.'

echo -e "\n2️⃣ Testing health..."
curl -s http://localhost:8000/health/ | jq '.status, .services'

echo -e "\n3️⃣ Testing text search..."
curl -s -X POST "http://localhost:8000/api/v2/search-text" \
     -H "Content-Type: application/json" \
     -d '{"query": "red nike shoes", "top_k": 2}' | \
     jq '.total_matches_found, .results[0].model_name // "No results"'

echo -e "\n✅ Quick test completed"