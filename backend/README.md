Nexora Backend (Milestone 2: Voice Transcription)

This FastAPI backend includes:
- Voice transcription via faster-whisper (STT)
- Audio processing endpoint

Milestone 3 adds:
- Intent extraction endpoint with strict JSON schema validation
- Optional Ollama support when `NEXORA_LLM_PROVIDER=ollama`

Setup:

1. Create a virtual environment: `python -m venv .venv`
2. Activate it: `.\.venv\Scripts\Activate.ps1` (PowerShell) or `.\.venv\Scripts\activate` (cmd)
3. Install: `pip install -r requirements.txt`
   (First run will download ~39MB tiny whisper model)
4. Run server: `uvicorn app.main:app --reload --host 0.0.0.0 --port 8005`

Endpoints:
- GET /ping → {"status":"ok","message":"pong"}
- GET /health → {"status":"healthy"}
- POST /transcribe → {"text":"...", "confidence":0.95, "language":"en", "status":"success"}
  (accepts: wav, mp3, m4a, flac, etc.)
- POST /extract-intent → structured intent JSON, or 422 if confidence is too low


