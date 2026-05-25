"""LLM-based translation adapter."""

from app.llm_router import OllamaLLMRouter
from app.routing.translation.base import TranslationEngine


class LLMTranslationEngine(TranslationEngine):
    """
    Sync wrapper placeholder for LLM translation.

    The main orchestrator currently uses HFTranslationEngine by default for
    deterministic behavior and lower latency/cost. This class is provided so
    callers can switch translation strategy without changing orchestration APIs.
    """

    def __init__(self, llm_router: OllamaLLMRouter):
        self.llm_router = llm_router

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        # Async LLM translation is handled by orchestration when explicitly enabled.
        # Returning empty keeps this adapter safe as an optional strategy.
        return ""
