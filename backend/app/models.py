from sqlalchemy import Column, DateTime, Float, Integer, String, Text, Boolean, JSON

from app.database import Base


class Task(Base):
	__tablename__ = "tasks"

	id = Column(Integer, primary_key=True, index=True)
	text = Column(String, nullable=False)
	intent = Column(String, nullable=False)
	department = Column(String, nullable=False)
	queue = Column(String, nullable=False)
	priority = Column(String, nullable=False, default="normal")
	status = Column(String, nullable=False, default="queued")
	confidence = Column(Float, nullable=False)
	raw_text = Column(String, nullable=False)
	source = Column(String, nullable=False)
	created_at = Column(DateTime, nullable=False)


class CallSession(Base):
	__tablename__ = "call_sessions"

	id = Column(Integer, primary_key=True, index=True)
	session_id = Column(String, unique=True, index=True, nullable=False)
	call_id = Column(String, index=True, nullable=False)
	language = Column(String, nullable=False, default="en")
	language_confidence = Column(Float, nullable=False, default=0.0)
	language_locked = Column(Boolean, nullable=False, default=False)
	state = Column(String, nullable=False, default="initiated")
	created_at = Column(DateTime, nullable=False)
	updated_at = Column(DateTime, nullable=False)
	ended_at = Column(DateTime, nullable=True)
	session_metadata = Column(JSON, name="metadata") # Keeps database column named 'metadata'
	transcription_history = Column(JSON, nullable=True, default=[])
	response_history = Column(JSON, nullable=True, default=[])
