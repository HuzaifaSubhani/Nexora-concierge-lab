"""Routing configuration maps and env-driven overrides."""

from __future__ import annotations

import json
import os
from typing import Any, Dict

DEFAULT_LANGUAGE_MODEL_MAP: Dict[str, str] = {
    "en": "qwen3:8b",
    "hi": "qwen3:8b",
    "th": "qwen3:8b",
    "ar": "qwen3:8b",
    "fr": "qwen3:8b",
    "it": "qwen3:8b",
    "es": "qwen3:8b",
    "de": "qwen3:8b",
    "nl": "qwen3:8b",
    "zh": "qwen3:8b",
    "ja": "qwen3:8b",
}

DEFAULT_LANGUAGE_GENERATION_MAP: Dict[str, Dict[str, Any]] = {
    "default": {"think": False, "temperature": 0.6, "top_k": 30, "top_p": 0.9, "num_predict": 48},
    "en": {"think": False, "temperature": 0.6, "top_k": 30, "top_p": 0.9, "num_predict": 48},
    "hi": {"think": False, "temperature": 0.55, "top_k": 30, "top_p": 0.9, "num_predict": 64},
    "th": {"think": False, "temperature": 0.5, "top_k": 30, "top_p": 0.9, "num_predict": 72},
    "ar": {"think": False, "temperature": 0.55, "top_k": 30, "top_p": 0.9, "num_predict": 64},
    "fr": {"think": False, "temperature": 0.6, "top_k": 30, "top_p": 0.9, "num_predict": 56},
    "it": {"think": False, "temperature": 0.6, "top_k": 30, "top_p": 0.9, "num_predict": 56},
    "es": {"think": False, "temperature": 0.6, "top_k": 30, "top_p": 0.9, "num_predict": 56},
    "de": {"think": False, "temperature": 0.6, "top_k": 30, "top_p": 0.9, "num_predict": 56},
    "nl": {"think": False, "temperature": 0.6, "top_k": 30, "top_p": 0.9, "num_predict": 56},
    "zh": {"think": False, "temperature": 0.5, "top_k": 30, "top_p": 0.9, "num_predict": 72},
    "ja": {"think": False, "temperature": 0.5, "top_k": 30, "top_p": 0.9, "num_predict": 72},
}

# Map language -> translation engine descriptor. The actual translation operation
# is handled by app.services.translate_text (HF first, then googletrans fallback).
DEFAULT_TRANSLATION_MODEL_MAP: Dict[str, str] = {
    "en": "direct",
    "hi": "Helsinki-NLP/opus-mt-hi-en",
    "th": "Helsinki-NLP/opus-mt-th-en",
    "ar": "Helsinki-NLP/opus-mt-ar-en",
    "fr": "Helsinki-NLP/opus-mt-fr-en",
    "it": "Helsinki-NLP/opus-mt-it-en",
    "es": "Helsinki-NLP/opus-mt-es-en",
    "de": "Helsinki-NLP/opus-mt-de-en",
    "nl": "Helsinki-NLP/opus-mt-nl-en",
    "zh": "Helsinki-NLP/opus-mt-zh-en",
    "ja": "Helsinki-NLP/opus-mt-ja-en",
}

# Per-language transcription profiles.
DEFAULT_TRANSCRIPTION_PROFILE_MAP: Dict[str, Dict[str, Any]] = {
    "default": {"model": "base", "language": None, "beam_size": 5, "vad_filter": True},
    "th": {"model": "base", "language": "th", "beam_size": 5, "vad_filter": False},
    "ar": {"model": "base", "language": "ar", "beam_size": 5, "vad_filter": True},
    "hi": {"model": "base", "language": "hi", "beam_size": 5, "vad_filter": True},
    "ja": {"model": "base", "language": "ja", "beam_size": 5, "vad_filter": True},
    "zh": {"model": "base", "language": "zh", "beam_size": 5, "vad_filter": True},
}


def _parse_json_env(env_name: str) -> Dict[str, Any]:
    raw = os.getenv(env_name, "").strip()
    if not raw:
        return {}
    try:
        value = json.loads(raw)
        if isinstance(value, dict):
            return value
    except Exception:
        return {}
    return {}


def get_effective_model_map() -> Dict[str, str]:
    model_map = dict(DEFAULT_LANGUAGE_MODEL_MAP)
    parsed = _parse_json_env("NEXORA_LLM_MODEL_MAP_JSON")
    for key, value in parsed.items():
        if isinstance(key, str) and isinstance(value, str) and key.strip() and value.strip():
            model_map[key.strip().lower()] = value.strip()
    return model_map


def get_effective_generation_map() -> Dict[str, Dict[str, Any]]:
    generation_map = {key: dict(value) for key, value in DEFAULT_LANGUAGE_GENERATION_MAP.items()}
    parsed = _parse_json_env("NEXORA_LLM_GENERATION_MAP_JSON")
    for key, value in parsed.items():
        if isinstance(key, str) and isinstance(value, dict):
            generation_map[key.strip().lower()] = dict(value)
    return generation_map


def get_effective_transcription_profile_map() -> Dict[str, Dict[str, Any]]:
    profile_map = {key: dict(value) for key, value in DEFAULT_TRANSCRIPTION_PROFILE_MAP.items()}
    parsed = _parse_json_env("NEXORA_STT_PROFILE_MAP_JSON")
    for key, value in parsed.items():
        if isinstance(key, str) and isinstance(value, dict):
            profile_map[key.strip().lower()] = dict(value)
    return profile_map


def get_supported_languages() -> list[str]:
    languages = set(get_effective_model_map().keys())
    languages.update(get_effective_generation_map().keys())
    languages.update(DEFAULT_TRANSLATION_MODEL_MAP.keys())
    languages.update(get_effective_transcription_profile_map().keys())
    languages.discard("default")
    return sorted(languages)
