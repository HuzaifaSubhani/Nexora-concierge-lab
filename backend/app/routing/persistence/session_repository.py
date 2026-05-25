"""Session repository adapter over SessionManager."""

from app.session_manager import CallState, SessionManager


class SessionRepository:
    """Encapsulates persistence-aware session mutations."""

    def __init__(self, manager: SessionManager):
        self.manager = manager

    def get(self, session_id: str):
        return self.manager.get_session(session_id)

    def transition(self, session_id: str, state: CallState):
        self.manager.update_session_state(session_id, state)

    def lock_language(self, session_id: str, language: str, confidence: float):
        self.manager.lock_session_language(session_id, language, confidence)

    def set_confidence(self, session_id: str, confidence: float):
        self.manager.set_session_language_confidence(session_id, confidence)

    def add_transcription(self, session_id: str, text: str, duration_ms: float = 0.0):
        self.manager.add_session_transcription(session_id, text, duration_ms=duration_ms)

    def add_response(self, session_id: str, text: str, source: str = "llm", latency_ms: float = 0.0):
        self.manager.add_session_response(session_id, text, source=source, latency_ms=latency_ms)

    def set_metadata_value(self, session_id: str, key: str, value):
        self.manager.set_session_metadata_value(session_id, key, value)

    def close(self, session_id: str, reason: str = "completed"):
        self.manager.close_session(session_id, reason)
