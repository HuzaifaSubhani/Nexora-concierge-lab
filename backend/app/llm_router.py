"""
LLM Routing and Inference Layer.
Routes transcribed text to Qwen3:8b via Ollama with language-aware prompts.
"""

import asyncio
import difflib
import json
import logging
import re
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from app.services import normalize_repeated_text

logger = logging.getLogger(__name__)


class OllamaLLMRouter:
    """Routes requests to Ollama-hosted Qwen3:8b LLM."""

    OLLAMA_BASE_URL = "http://localhost:11434"
    MODEL_NAME = "qwen3:8b"
    DEFAULT_TIMEOUT = 180.0

    def __init__(self, base_url: str = OLLAMA_BASE_URL, model: str = MODEL_NAME):
        self.base_url = base_url
        self.model = model

    def _request_json(
        self,
        method: str,
        path: str,
        payload: Optional[Dict[str, Any]] = None,
        timeout: float = 30.0,
    ) -> Dict[str, Any]:
        data = None
        headers = {"Content-Type": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")

        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers=headers,
            method=method,
        )

        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            if not body.strip():
                return {}
            return json.loads(body)

    async def health_check(self) -> bool:
        """Check if Ollama service is running and the configured model is available."""
        try:
            data = await asyncio.to_thread(self._request_json, "GET", "/api/tags", None, 5.0)
            models = [m.get("name", "").split(":")[0] for m in data.get("models", [])]
            model_available = any(self.model.split(":")[0] in name for name in models)

            if not model_available:
                logger.warning(f"Model {self.model} not found in Ollama. Available: {models}")
                return False

            logger.info(f"Ollama health check passed. Model {self.model} available.")
            return True
        except Exception as exc:
            logger.error(f"Ollama health check error: {exc}")
            return False

    async def generate_response(
        self,
        transcribed_text: str,
        language: str,
        system_prompt: str,
        model_override: Optional[str] = None,
        generation_profile: Optional[Dict[str, Any]] = None,
        stream: bool = False,
    ) -> Optional[str]:
        """Generate an LLM response for the provided transcript."""
        try:
            selected_model = model_override or self.model
            effective_profile: Dict[str, Any] = {
                "think": False,
                "temperature": 0.7,
                "top_k": 40,
                "top_p": 0.9,
                "num_predict": 48,
            }
            if isinstance(generation_profile, dict):
                for key in ("think", "temperature", "top_k", "top_p", "num_predict"):
                    if key in generation_profile:
                        effective_profile[key] = generation_profile[key]

            payload = {
                "model": selected_model,
                "system": system_prompt,
                "prompt": transcribed_text,
                "stream": False,
                "think": bool(effective_profile.get("think", False)),
                "temperature": float(effective_profile.get("temperature", 0.7)),
                "top_k": int(effective_profile.get("top_k", 40)),
                "top_p": float(effective_profile.get("top_p", 0.9)),
                "options": {
                    "num_predict": int(effective_profile.get("num_predict", 48)),
                },
            }

            def extract_candidate(body_text: str) -> Optional[str]:
                parsed = None
                try:
                    parsed = json.loads(body_text)
                except Exception:
                    parsed = None

                if isinstance(parsed, dict):
                    for key in ("response", "text", "result", "output", "generated_text"):
                        value = parsed.get(key)
                        if isinstance(value, str) and value.strip():
                            return normalize_repeated_text(value.strip())

                    for key in ("choices", "results", "outputs"):
                        value = parsed.get(key)
                        if isinstance(value, list) and value:
                            first = value[0]
                            if isinstance(first, dict):
                                for subkey in ("text", "message", "content", "response"):
                                    subvalue = first.get(subkey)
                                    if isinstance(subvalue, str) and subvalue.strip():
                                        return normalize_repeated_text(subvalue.strip())
                            elif isinstance(first, str) and first.strip():
                                return normalize_repeated_text(first.strip())

                if body_text.strip():
                    return normalize_repeated_text(body_text.strip())

                return None

            logger.info(f"LLM route selected model='{selected_model}' for language='{language}'")
            logger.info(f"LLM generation profile: {effective_profile}")
            logger.debug(f"Ollama payload: {json.dumps(payload)[:1000]}")

            try:
                response_data = await asyncio.to_thread(
                    self._request_json,
                    "POST",
                    "/api/generate",
                    payload,
                    self.DEFAULT_TIMEOUT,
                )
            except urllib.error.HTTPError as exc:
                logger.error(f"Ollama inference failed: {exc.code} - {exc.reason}")
                return None

            final_text = extract_candidate(json.dumps(response_data))
            if not final_text:
                return None

            try:
                ratio = difflib.SequenceMatcher(
                    None,
                    re.sub(r"\W+", " ", transcribed_text.lower()).strip(),
                    re.sub(r"\W+", " ", final_text.lower()).strip(),
                ).ratio()
            except Exception:
                ratio = 1.0

            if ratio < 0.18:
                logger.warning(
                    f"Possible hallucination detected (sim={ratio:.2f}). Retrying with deterministic settings."
                )
                retry_payload = dict(payload)
                retry_payload["temperature"] = 0.0
                retry_payload["think"] = False
                retry_payload["options"] = dict(retry_payload.get("options", {}))
                retry_payload["options"]["num_predict"] = max(
                    16,
                    int(retry_payload.get("options", {}).get("num_predict", 48) // 2),
                )

                try:
                    retry_data = await asyncio.to_thread(
                        self._request_json,
                        "POST",
                        "/api/generate",
                        retry_payload,
                        self.DEFAULT_TIMEOUT,
                    )
                    retry_text = extract_candidate(json.dumps(retry_data))
                    if retry_text:
                        return retry_text
                except urllib.error.HTTPError:
                    pass

            return final_text

        except Exception as exc:
            logger.error(f"LLM generation error: {exc}")
            return None

    async def list_available_models(self) -> List[str]:
        """List models exposed by the local Ollama instance."""
        try:
            data = await asyncio.to_thread(self._request_json, "GET", "/api/tags", None, 10.0)
            models = data.get("models", [])
            out: List[str] = []
            for item in models:
                name = item.get("name", "")
                if isinstance(name, str) and name.strip():
                    out.append(name.strip())
            return out
        except Exception as exc:
            logger.warning(f"Unable to list models from Ollama: {exc}")
            return []


_llm_router: Optional[OllamaLLMRouter] = None


def get_llm_router() -> OllamaLLMRouter:
    """Return a shared router instance."""
    global _llm_router
    if _llm_router is None:
        _llm_router = OllamaLLMRouter()
    return _llm_router
