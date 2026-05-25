"""Text language detection adapter."""

from app.language_router import LanguageRouter, get_language_router
from app.routing.contracts import LanguageEvidence


class TextLanguageDetector:
    """Bridge text detection results into routing contracts."""

    def __init__(self, router: LanguageRouter | None = None):
        self.router = router or get_language_router()

    def detect(self, text: str) -> LanguageEvidence:
        language, confidence, should_lock = self.router.detect_language_from_transcription(text)
        return LanguageEvidence(
            language=language,
            confidence=confidence,
            source="text_langdetect",
            lock_recommended=should_lock,
        )
