"""Translation engine using services.translate_text (HF + fallbacks)."""

from app.routing.translation.base import TranslationEngine
from app.services import translate_text


class HFTranslationEngine(TranslationEngine):
    """HuggingFace-first translation path with existing fallbacks."""

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        return translate_text(text, src_lang, tgt_lang)
