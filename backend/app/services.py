"""
Transcription and AI services for Nexora.
"""
import os
import tempfile

WhisperModel = None
_whisper_import_error = None

try:
    from faster_whisper import WhisperModel as FasterWhisperModel
    WhisperModel = FasterWhisperModel
except Exception as exc:  # pragma: no cover - environment-specific dependency issue
    _whisper_import_error = exc

# Load faster-whisper model (tiny for speed, base for accuracy)
# tiny = ~39M, base = ~140M
_model = None


def get_whisper_model():
    """Lazy-load whisper model."""
    global _model
    if _model is None:
        if WhisperModel is None:
            raise RuntimeError(
                "faster-whisper is not available in this environment. "
                "Install backend requirements in a Python 3.10 venv."
            ) from _whisper_import_error
        print("[INFO] Loading faster-whisper model (tiny)...")
        _model = WhisperModel("tiny", device="cpu", compute_type="int8")
    return _model


async def transcribe_audio(audio_data: bytes) -> dict:
    """
    Transcribe audio bytes to text using faster-whisper.
    
    Args:
        audio_data: Raw audio bytes (WAV, MP3, etc.)
    
    Returns:
        {"text": "...", "confidence": 0.95, "language": "en"}
    """
    try:
        model = get_whisper_model()
        
        # Write audio to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_data)
            tmp_path = tmp.name
        
        print(f"[INFO] Transcribing audio file: {tmp_path} ({len(audio_data)} bytes)")
        
        try:
            # Transcribe and unpack the returned (segments, info) pair.
            segments, info = model.transcribe(tmp_path, language="en", beam_size=5)
            segments_list = list(segments)
            
            print(f"[INFO] Got {len(segments_list)} segments")
            
            # Build text from segment text and derive a bounded confidence score.
            text = " ".join([s.text for s in segments_list])
            confidences = []
            for segment in segments_list:
                no_speech_prob = getattr(segment, "no_speech_prob", None)
                if no_speech_prob is not None:
                    confidences.append(max(0.0, min(1.0, 1.0 - float(no_speech_prob))))
            if confidences:
                confidence = sum(confidences) / len(confidences)
            else:
                confidence = max(0.0, min(1.0, float(getattr(info, "language_probability", 0.0) or 0.0)))
            
            print(f"[INFO] Transcription complete: {len(text)} chars, confidence: {confidence}")
            
            return {
                "text": text.strip(),
                "confidence": round(confidence, 3),
                "language": "en",
                "status": "success"
            }
        finally:
            # Clean up temp file
            os.unlink(tmp_path)
    
    except Exception as e:
        print(f"[ERROR] Transcription failed: {str(e)}")
        return {
            "text": "",
            "error": str(e),
            "status": "error"
        }
