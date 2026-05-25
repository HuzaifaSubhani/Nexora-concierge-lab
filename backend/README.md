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
4. Run server: `\.\run.ps1`

Optional: medium multilingual + int8 CPU (without changing default behavior)

- Keep defaults as-is (current implementation):
  - `NEXORA_WHISPER_MODEL=tiny`
  - `NEXORA_WHISPER_LANGUAGE=en`
  - `NEXORA_WHISPER_DEVICE=cpu`
  - `NEXORA_WHISPER_COMPUTE_TYPE=int8`
- For multilingual medium model:
  - `NEXORA_WHISPER_MODEL=medium`
  - `NEXORA_WHISPER_LANGUAGE=auto`
  - `NEXORA_WHISPER_DEVICE=cpu`
  - `NEXORA_WHISPER_COMPUTE_TYPE=int8`

PowerShell example:
- `$env:NEXORA_WHISPER_MODEL='medium'`
- `$env:NEXORA_WHISPER_LANGUAGE='auto'`
- `$env:NEXORA_WHISPER_DEVICE='cpu'`
- `$env:NEXORA_WHISPER_COMPUTE_TYPE='int8'`
- `$env:NEXORA_WHISPER_PRELOAD='true'`
- `\.\run.ps1`

Warmup options:
- Startup preload: set `NEXORA_WHISPER_PRELOAD=true` to load model during API startup.
- On-demand warmup: call `POST /transcribe/warmup` before first transcription.

Endpoints:
- GET /ping → {"status":"ok","message":"pong"}
- GET /health → {"status":"healthy"}
- POST /transcribe → {"text":"...", "confidence":0.95, "language":"en", "status":"success"}
  (accepts: wav, mp3, m4a, flac, etc.)
- POST /extract-intent → structured intent JSON, or 422 if confidence is too low


