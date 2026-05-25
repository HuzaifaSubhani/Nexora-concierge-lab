from app.language_router import get_language_router


def test_detect_english():
    router = get_language_router()
    text = "Hello, I would like to book a room for two nights starting next Friday."
    lang, confidence, should_lock = router.detect_language_from_transcription(text)
    assert lang == "en"
    assert confidence >= 50.0


def test_short_text_low_confidence():
    router = get_language_router()
    text = "Hi"
    lang, confidence, should_lock = router.detect_language_from_transcription(text)
    # For very short text we expect low confidence and no lock
    assert isinstance(lang, str)
    assert confidence < router.CONFIDENCE_THRESHOLD
    assert not should_lock
