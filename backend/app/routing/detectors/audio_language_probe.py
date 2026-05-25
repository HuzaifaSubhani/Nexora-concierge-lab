"""Audio language probe adapter."""

from app.routing.contracts import LanguageEvidence
from app.services import probe_audio_language


class AudioLanguageProbe:
    """First-pass audio language probe used for STT profile routing."""

    def probe(self, audio_data: bytes) -> LanguageEvidence:
        result = probe_audio_language(audio_data)
        language = str(result.get("language") or "unknown").strip().lower()
        confidence = float(result.get("confidence", 0.0) or 0.0) * 100.0
        return LanguageEvidence(
            language=language if language else "unknown",
            confidence=max(0.0, min(100.0, confidence)),
            source=str(result.get("source") or "whisper_probe"),
            lock_recommended=confidence >= 70.0,
        )
