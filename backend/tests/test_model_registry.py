from app.routing.model_registry import (
    is_model_available,
    resolve_generation_profile_for_language,
    resolve_model_for_language,
    resolve_transcription_profile_for_language,
    resolve_translation_model_for_language,
)


def test_model_availability_base_match():
    assert is_model_available("qwen3:8b", ["qwen3:latest"]) is True
    assert is_model_available("llama3.2:3b", ["qwen3:8b"]) is False


def test_resolve_model_with_fallback():
    decision = resolve_model_for_language("en", default_model="qwen3:8b", available_models=["phi4:mini"])
    assert decision.requested_model
    assert decision.selected_model == "qwen3:8b"
    assert decision.model_fallback_used is True


def test_resolve_transcription_profile_defaults():
    profile = resolve_transcription_profile_for_language("xx")
    assert profile.model
    assert isinstance(profile.beam_size, int)
    assert isinstance(profile.vad_filter, bool)


def test_resolve_generation_profile():
    profile = resolve_generation_profile_for_language("th")
    assert "num_predict" in profile
    assert "temperature" in profile


def test_resolve_translation_model():
    assert resolve_translation_model_for_language("en") == "direct"
    assert resolve_translation_model_for_language("it") == "Helsinki-NLP/opus-mt-it-en"
    assert resolve_translation_model_for_language("nl") == "Helsinki-NLP/opus-mt-nl-en"
    assert resolve_translation_model_for_language("unknown_lang") == "auto"
