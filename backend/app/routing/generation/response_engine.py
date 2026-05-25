"""LLM response generation adapter."""

from typing import Any, Dict, Optional

from app.llm_router import OllamaLLMRouter


class ResponseGenerationEngine:
    """Thin wrapper around Ollama router for orchestration portability."""

    def __init__(self, llm_router: OllamaLLMRouter):
        self.llm_router = llm_router

    async def generate(
        self,
        text: str,
        language: str,
        system_prompt: str,
        model_override: Optional[str] = None,
        generation_profile: Optional[Dict[str, Any]] = None,
        stream: bool = False,
    ):
        return await self.llm_router.generate_response(
            transcribed_text=text,
            language=language,
            system_prompt=system_prompt,
            model_override=model_override,
            generation_profile=generation_profile,
            stream=stream,
        )
