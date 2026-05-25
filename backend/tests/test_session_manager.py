import time
from app.session_manager import SessionManager, CallState


def test_session_create_and_persist():
    mgr = SessionManager()
    session = mgr.create_session(call_id="test_call_123")
    assert session is not None
    assert session.call_id == "test_call_123"

    # Lock language and persist
    mgr.lock_session_language(session.session_id, "en", 95.0)
    # Give small delay for DB commit
    time.sleep(0.1)

    # Query DB to ensure record exists
    db = mgr._SessionLocal()
    try:
        rec = db.query(mgr._CallSessionModel).filter(mgr._CallSessionModel.session_id == session.session_id).first()
        assert rec is not None
        assert rec.language == "en"
        assert rec.language_locked is True

        # Verify persistence-aware helper methods keep DB in sync
        mgr.add_session_transcription(session.session_id, "Need towels for room 502")
        mgr.add_session_response(session.session_id, "Sure, we will send towels now.", latency_ms=321.0)
        mgr.set_session_language_confidence(session.session_id, 97.5)

        rec = db.query(mgr._CallSessionModel).filter(mgr._CallSessionModel.session_id == session.session_id).first()
        assert rec is not None
        assert isinstance(rec.transcription_history, list) and len(rec.transcription_history) >= 1
        assert isinstance(rec.response_history, list) and len(rec.response_history) >= 1
        assert rec.language_confidence == 97.5
    finally:
        # Cleanup
        if rec:
            db.delete(rec)
            db.commit()
        db.close()
