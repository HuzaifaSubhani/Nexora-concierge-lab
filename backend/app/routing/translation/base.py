"""Base contract for translation engines."""

from abc import ABC, abstractmethod


class TranslationEngine(ABC):
    """Abstract translation engine contract."""

    @abstractmethod
    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        raise NotImplementedError
