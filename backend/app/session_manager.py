"""
Session Management for Call Routing.
Tracks call state, language locks, and persistence to database.
"""
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)

class CallState(str, Enum):
    """Call session states."""
    INITIATED = "initiated"
    LANGUAGE_DETECTING = "language_detecting"
    LANGUAGE_LOCKED = "language_locked"
    ROUTING = "routing"
    IN_CONVERSATION = "in_conversation"
    ENDED = "ended"
    FAILED = "failed"


class CallSession:
    """
    In-memory representation of a call session.
    Mirrors database model for persistence layer.
    """
    
    def __init__(
        self,
        session_id: Optional[str] = None,
        call_id: str = None,
        language: Optional[str] = None,
        language_confidence: float = 0.0,
        language_locked: bool = False,
        state: CallState = CallState.INITIATED,
    ):
        self.session_id = session_id or str(uuid.uuid4())
        self.call_id = call_id or f"call_{uuid.uuid4().hex[:8]}"
        self.language = language or "en"
        self.language_confidence = language_confidence
        self.language_locked = language_locked
        self.state = state
        self.created_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)
        self.ended_at: Optional[datetime] = None
        self.metadata: Dict[str, Any] = {}
        self.transcription_history: list = []
        self.response_history: list = []
    
    def lock_language(self, language: str, confidence: float = 100.0):
        """Lock language for remainder of session."""
        self.language = language
        self.language_confidence = confidence
        self.language_locked = True
        self.state = CallState.LANGUAGE_LOCKED
        self.updated_at = datetime.now(timezone.utc)
        logger.info(f"[Session {self.session_id}] Language locked: {language} ({confidence:.1f}%)")
    
    def transition_state(self, new_state: CallState):
        """Transition to new call state."""
        old_state = self.state
        self.state = new_state
        self.updated_at = datetime.now(timezone.utc)
        logger.info(f"[Session {self.session_id}] State transition: {old_state} -> {new_state}")
    
    def add_transcription(self, text: str, duration_ms: float = 0.0):
        """Record transcribed user input."""
        self.transcription_history.append({
            "text": text,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "duration_ms": duration_ms,
        })
    
    def add_response(self, text: str, source: str = "llm", latency_ms: float = 0.0):
        """Record LLM response."""
        self.response_history.append({
            "text": text,
            "source": source,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "latency_ms": latency_ms,
        })
    
    def end_session(self, reason: str = "completed"):
        """Mark session as ended."""
        self.state = CallState.ENDED
        self.ended_at = datetime.now(timezone.utc)
        self.metadata["end_reason"] = reason
        self.updated_at = self.ended_at
        logger.info(f"[Session {self.session_id}] Ended. Reason: {reason}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary for JSON serialization."""
        return {
            "session_id": self.session_id,
            "call_id": self.call_id,
            "language": self.language,
            "language_confidence": self.language_confidence,
            "language_locked": self.language_locked,
            "state": self.state.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "metadata": self.metadata,
            "transcription_count": len(self.transcription_history),
            "response_count": len(self.response_history),
        }


class SessionManager:
    """
    Manages in-memory call sessions with database persistence hooks.
    """
    
    def __init__(self):
        self._sessions: Dict[str, CallSession] = {}
        # Use SessionLocal for persistence
        from app.database import SessionLocal
        from app.models import CallSession as CallSessionModel
        self._SessionLocal = SessionLocal
        self._CallSessionModel = CallSessionModel
    
    def create_session(self, call_id: Optional[str] = None) -> CallSession:
        """Create new call session."""
        session = CallSession(call_id=call_id)
        self._sessions[session.session_id] = session
        logger.info(f"Created session: {session.session_id} for call: {session.call_id}")
        # Persist to DB
        try:
            self._persist_session(session)
        except Exception as e:
            logger.warning(f"Failed to persist session on create: {e}")
        return session
    
    def get_session(self, session_id: str) -> Optional[CallSession]:
        """Retrieve session by ID."""
        return self._sessions.get(session_id)
    
    def update_session_state(self, session_id: str, new_state: CallState):
        """Update session state and persist to DB."""
        session = self.get_session(session_id)
        if session:
            session.transition_state(new_state)
            try:
                self._persist_session(session)
            except Exception as e:
                logger.warning(f"Failed to persist session state update: {e}")

    def set_session_language_confidence(self, session_id: str, confidence: float):
        """Update language confidence and persist."""
        session = self.get_session(session_id)
        if session:
            session.language_confidence = confidence
            session.updated_at = datetime.now(timezone.utc)
            try:
                self._persist_session(session)
            except Exception as e:
                logger.warning(f"Failed to persist language confidence update: {e}")
    
    def lock_session_language(self, session_id: str, language: str, confidence: float = 100.0):
        """Lock language for session and persist."""
        session = self.get_session(session_id)
        if session:
            session.lock_language(language, confidence)
            try:
                self._persist_session(session)
            except Exception as e:
                logger.warning(f"Failed to persist language lock: {e}")

    def add_session_transcription(self, session_id: str, text: str, duration_ms: float = 0.0):
        """Append transcription history entry and persist."""
        session = self.get_session(session_id)
        if session:
            session.add_transcription(text, duration_ms=duration_ms)
            session.updated_at = datetime.now(timezone.utc)
            try:
                self._persist_session(session)
            except Exception as e:
                logger.warning(f"Failed to persist transcription append: {e}")

    def add_session_response(self, session_id: str, text: str, source: str = "llm", latency_ms: float = 0.0):
        """Append response history entry and persist."""
        session = self.get_session(session_id)
        if session:
            session.add_response(text, source=source, latency_ms=latency_ms)
            session.updated_at = datetime.now(timezone.utc)
            try:
                self._persist_session(session)
            except Exception as e:
                logger.warning(f"Failed to persist response append: {e}")

    def set_session_metadata_value(self, session_id: str, key: str, value: Any):
        """Set one metadata key and persist."""
        session = self.get_session(session_id)
        if session:
            session.metadata[key] = value
            session.updated_at = datetime.now(timezone.utc)
            try:
                self._persist_session(session)
            except Exception as e:
                logger.warning(f"Failed to persist session metadata update: {e}")
    
    def close_session(self, session_id: str, reason: str = "completed"):
        """Close session and persist to database."""
        session = self.get_session(session_id)
        if session:
            session.end_session(reason)
            try:
                self._persist_session(session)
            except Exception as e:
                logger.warning(f"Failed to persist closed session: {e}")
            logger.info(f"Closed session: {session_id}")

    def persist_session(self, session_id: str):
        """Persist current in-memory state for a session."""
        session = self.get_session(session_id)
        if session:
            try:
                self._persist_session(session)
            except Exception as e:
                logger.warning(f"Failed to persist session: {e}")
    
    def get_active_sessions_count(self) -> int:
        """Get count of active sessions."""
        return sum(1 for s in self._sessions.values() if s.state != CallState.ENDED)
    
    def cleanup_old_sessions(self, max_age_minutes: int = 120):
        """
        Remove sessions older than max_age.
        Useful for in-memory cleanup of completed calls.
        """
        now = datetime.now(timezone.utc)
        sessions_to_remove = []
        
        for session_id, session in self._sessions.items():
            age_minutes = (now - session.updated_at).total_seconds() / 60
            if age_minutes > max_age_minutes and session.state == CallState.ENDED:
                sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            del self._sessions[session_id]
            logger.info(f"Cleaned up old session: {session_id}")
        
        return len(sessions_to_remove)

    def _persist_session(self, session: CallSession):
        """Persist an in-memory session into the database (upsert)."""
        db = self._SessionLocal()
        try:
            # Attempt to find existing record
            existing = db.query(self._CallSessionModel).filter(self._CallSessionModel.session_id == session.session_id).first()
            if existing is None:
                record = self._CallSessionModel(
                    session_id=session.session_id,
                    call_id=session.call_id,
                    language=session.language,
                    language_confidence=session.language_confidence,
                    language_locked=session.language_locked,
                    state=session.state.value,
                    created_at=session.created_at,
                    updated_at=session.updated_at,
                    ended_at=session.ended_at,
                    metadata=session.metadata,
                    transcription_history=session.transcription_history,
                    response_history=session.response_history,
                )
                db.add(record)
            else:
                existing.call_id = session.call_id
                existing.language = session.language
                existing.language_confidence = session.language_confidence
                existing.language_locked = session.language_locked
                existing.state = session.state.value
                existing.updated_at = session.updated_at
                existing.ended_at = session.ended_at
                existing.metadata = session.metadata
                existing.transcription_history = session.transcription_history
                existing.response_history = session.response_history

            db.commit()
        finally:
            db.close()


# Global singleton instance
_manager_instance = None

def get_session_manager() -> SessionManager:
    """Get or create singleton session manager."""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = SessionManager()
    return _manager_instance
