from app.routing.orchestrator import get_voice_routing_orchestrator
from app.session_manager import get_session_manager


def test_orchestrator_detect_language_updates_session():
    manager = get_session_manager()
    orchestrator = get_voice_routing_orchestrator()
    session = manager.create_session(call_id="orchestrator_test_call")

    try:
        result = orchestrator.detect_language_for_session(
            session_id=session.session_id,
            transcription="Hello, I need room service please.",
        )
        assert result["detected_language"] == "en"
        assert result["state"] == "LANGUAGE_CONFIRMATION"
        assert result["action"] == "ask_language_confirmation"
        assert result["confidence"] >= 50.0
        assert result["language_locked"] is False
        assert bool(result.get("confirmation_prompt")) is True

        updated = manager.get_session(session.session_id)
        assert updated is not None
        assert updated.language_locked is False
        assert updated.metadata.get("pending_language") == "en"
        assert len(updated.transcription_history) >= 1

        confirm = orchestrator.confirm_language_for_session(session.session_id, confirm=True)
        assert confirm["state"] == "LISTENING_TO_QUERY"
        assert confirm["action"] == "ask_query"
        assert confirm["language_locked"] is True
        assert confirm["selected_language"] == "en"
    finally:
        # Clean up persisted record
        db = manager._SessionLocal()
        try:
            rec = (
                db.query(manager._CallSessionModel)
                .filter(manager._CallSessionModel.session_id == session.session_id)
                .first()
            )
            if rec:
                db.delete(rec)
                db.commit()
        finally:
            db.close()
