"""Decision helpers for routing choices."""

from app.routing.contracts import LanguageEvidence


def merge_language_evidence(audio: LanguageEvidence | None, text: LanguageEvidence | None) -> LanguageEvidence | None:
    """
    Merge audio/text evidence, preferring higher confidence evidence when both exist.
    """
    if audio is None and text is None:
        return None
    if audio is None:
        return text
    if text is None:
        return audio
    return audio if audio.confidence >= text.confidence else text
