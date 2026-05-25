"""Whisper-backed transcription engine adapter."""

from typing import Any, Dict, Optional

from app.routing.contracts import TranscriptionProfile, TranscriptionResult
from app.routing.transcription.base import TranscriptionEngine
from app.services import transcribe_audio


class WhisperTranscriptionEngine(TranscriptionEngine):
    """Adapter around app.services.transcribe_audio with normalized output."""

    async def transcribe(
        self,
        audio_data: bytes,
        profile: Optional[Dict[str, Any]] = None,
        probe_result: Optional[Dict[str, Any]] = None,
    ) -> TranscriptionResult:
        raw = await transcribe_audio(audio_data, profile=profile, probe_result=probe_result)
        raw_profile = raw.get("transcription_profile", {}) if isinstance(raw, dict) else {}
        mapped_profile = None
        if raw_profile:
            mapped_profile = TranscriptionProfile(
                model=str(raw_profile.get("model", "base")),
                language=raw_profile.get("language"),
                beam_size=int(raw_profile.get("beam_size", 5)),
                vad_filter=bool(raw_profile.get("vad_filter", True)),
            )

        return TranscriptionResult(
            text=raw.get("text", ""),
            confidence=float(raw.get("confidence", 0.0) or 0.0),
            language=str(raw.get("language", "unknown")),
            status=str(raw.get("status", "error")),
            segments=int(raw.get("segments", 0) or 0),
            translation=raw.get("translation"),
            translation_language=raw.get("translation_language"),
            profile=mapped_profile,
        )
