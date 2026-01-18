```bash
docker build -t pocket_tts_api .
docker run --name pocket_tts_api -d -p 8001:8000 pocket_tts_api
```

Test server with cURL:

```bash
curl http://localhost:8001/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "Hello! This is a test of the fully compatible local text to speech server.",
    "voice": "nova",
    "response_format":"wav",
    "speed": 1.1
  }' \
  --output test_audio.wav
  ```