"""Model/profile resolution helpers for the routing layer."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.routing.config import (
    DEFAULT_TRANSLATION_MODEL_MAP,
    get_effective_generation_map,
    get_effective_model_map,
    get_effective_transcription_profile_map,
)
from app.routing.contracts import ModelDecision, TranscriptionProfile


def is_model_available(target_model: str, available_models: List[str]) -> bool:
    if not target_model:
        return False
    target = target_model.strip().lower()
    target_base = target.split(":")[0]
    for model in available_models:
        model_name = (model or "").strip().lower()
        if not model_name:
            continue
        if model_name == target:
            return True
        if model_name.split(":")[0] == target_base:
            return True
    return False


def resolve_translation_model_for_language(language: str) -> str:
    lang_key = (language or "").strip().lower()
    return DEFAULT_TRANSLATION_MODEL_MAP.get(lang_key, "auto")


def resolve_transcription_profile_for_language(language: Optional[str]) -> TranscriptionProfile:
    profiles = get_effective_transcription_profile_map()
    key = (language or "").strip().lower()
    selected = profiles.get(key, profiles.get("default", {}))
    return TranscriptionProfile(
        model=str(selected.get("model", "base")),
        language=selected.get("language"),
        beam_size=int(selected.get("beam_size", 5)),
        vad_filter=bool(selected.get("vad_filter", True)),
    )


def resolve_generation_profile_for_language(language: str) -> Dict[str, Any]:
    generation_map = get_effective_generation_map()
    key = (language or "").strip().lower()
    if key in generation_map and isinstance(generation_map[key], dict):
        return dict(generation_map[key])
    return dict(generation_map.get("default", {}))


def resolve_model_for_language(language: str, default_model: str, available_models: Optional[List[str]] = None) -> ModelDecision:
    model_map = get_effective_model_map()
    requested_model = model_map.get((language or "en").lower(), default_model)
    selected_model = requested_model
    model_fallback_used = False

    if available_models and not is_model_available(requested_model, available_models):
        selected_model = default_model
        model_fallback_used = True

    return ModelDecision(
        requested_model=requested_model,
        selected_model=selected_model,
        model_fallback_used=model_fallback_used,
        generation_profile=resolve_generation_profile_for_language(language),
    )
