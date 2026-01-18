# Pocket TTS API

Lightweight local TTS server based on the very fast [Pocket TTS model](https://kyutai.org/blog/2026-01-13-pocket-tts) from Kyutai, provides a simple OpenAI-compatible speech API (`v1/audio/speech`) for generating audio from text.

Using an old Haswell CPU it generated around 1.5x real-time speed for 24 KHz audio with the `nova` voice.

Inspired by [kyutai-tts-openai-api](kyutai-tts-openai-api).

# Build and run with Docker:

```bash
docker build -t pocket_tts_api .
docker run --name pocket_tts_api -d -p 8008:8000 pocket_tts_api
```

Currently the `model` and `speed` parameters are ignored.

# Test server with `curl`:

```bash
curl http://localhost:8008/v1/audio/speech \
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