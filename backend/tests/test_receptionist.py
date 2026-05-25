from app.receptionist import (
    confirmation_prompt_for_language,
    detect_supported_language,
    english_meaning_from_keywords,
    language_code_from_name_or_code,
    parse_confirmation,
    route_receptionist_query,
)


def test_language_selection_and_confirmation_words():
    assert language_code_from_name_or_code("French") == "fr"
    assert language_code_from_name_or_code("espanol") == "es"
    assert parse_confirmation("oui", "fr") is True
    assert parse_confirmation("nein", "de") is False
    assert confirmation_prompt_for_language("de").startswith("Ich habe Deutsch")
    assert "d\u00e9tect\u00e9" in confirmation_prompt_for_language("fr")
    assert detect_supported_language("Bonjour, je veux prendre rendez-vous.").code == "fr"


def test_route_appointment_from_french_meaning():
    english = english_meaning_from_keywords("Je veux prendre rendez-vous.", "fr")
    decision = route_receptionist_query(
        caller_text_original="Je veux prendre rendez-vous.",
        caller_text_english=english,
        language_code="fr",
    )

    assert decision.intent == "Appointments"
    assert decision.confidence == "high"
    assert decision.action == "route_department"
    assert "rendez-vous" in decision.response_final_language.lower()


def test_unclear_repeats_go_to_human():
    decision = route_receptionist_query(
        caller_text_original="I need something",
        caller_text_english="I need something",
        language_code="en",
        failure_count=2,
    )

    assert decision.intent == "Human Transfer"
    assert decision.action == "transfer_human"


def test_repeated_no_speech_goes_to_human():
    decision = route_receptionist_query(
        caller_text_original="",
        caller_text_english="",
        language_code="es",
        failure_count=2,
    )

    assert decision.intent == "Human Transfer"
    assert decision.action == "transfer_human"
    assert "representante humano" in decision.response_final_language.lower()
