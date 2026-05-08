from sqlalchemy import Column, DateTime, Float, Integer, String

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
