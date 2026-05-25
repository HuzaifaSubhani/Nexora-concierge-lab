Nexora — Milestone-driven AI Receptionist

## Milestone 2: Voice Transcription (STT)

**Backend**: faster-whisper integration + /transcribe endpoint
**Frontend**: Mic capture + audio submission

Quick start (PowerShell):

# Backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Frontend (separate shell)
cd frontend
npm run dev

## Testing

1. Open http://localhost:3000
2. Verify backend status shows pong
3. Click "Start Recording"
4. Speak clearly (e.g., "I need a towel")
5. Click "Stop"
6. See transcription JSON with text, confidence, language

Backend runs on http://localhost:8000
Frontend runs on http://localhost:3000

## Note
First transcription may take 30-60s (faster-whisper downloads tiny model ~39MB)

## Milestone 3: LLM Intent Extraction

1. Paste or record a request like `I need a towel`
2. Click `Extract Intent`
3. Confirm the returned JSON includes `intent`, `department`, `items`, `quantity`, and `confidence`

If confidence is below `0.6`, the backend returns a clarification/human-handoff response.
