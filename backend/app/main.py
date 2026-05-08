from datetime import datetime, timezone
from typing import List, Optional

from fastapi import Depends, FastAPI, UploadFile, File, HTTPException
from fastapi.params import Depends as DependsParam
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.database import Base, SessionLocal, engine, get_db
from app.intents import extract_intent
from app.models import Task
from app.schemas import AllowedDepartment, IntentExtractionRequest, IntentExtractionResult, TaskCreateRequest, TaskRecord, TaskRouteRequest, TaskRouteResult
from app.services import transcribe_audio

app = FastAPI(title="Nexora Backend - Milestone 2: Voice Transcription")

Base.metadata.create_all(bind=engine)

DEPARTMENT_QUEUES = {
    "housekeeping": "housekeeping_queue",
    "kitchen": "kitchen_queue",
    "maintenance": "maintenance_queue",
    "front_desk": "front_desk_queue",
    "room_service": "room_service_queue",
}


def _resolve_db_session(db: Session):
    if isinstance(db, DependsParam) or not hasattr(db, "add"):
        return SessionLocal(), True
    return db, False

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:3005",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
        "http://127.0.0.1:3005",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/ping")
async def ping():
    return {"status": "ok", "message": "pong"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """
    Transcribe audio file to text.
    
    Accepts: wav, mp3, m4a, flac, etc.
    Returns: {"text": "...", "confidence": 0.95, "language": "en", "status": "success"}
    """
    audio_data = await file.read()
    result = await transcribe_audio(audio_data)
    return result


@app.post("/extract-intent", response_model=IntentExtractionResult)
async def extract_intent_route(payload: IntentExtractionRequest):
    result = extract_intent(payload.text)
    if result.confidence < 0.6:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Low confidence intent. Ask for clarification or hand off to human.",
                "result": result.dict(),
            },
        )
    return result


@app.post("/route-task", response_model=TaskRouteResult)
async def route_task_route(payload: TaskRouteRequest):
    return _route_text(payload.text)


def _route_text(text: str) -> TaskRouteResult:
    result = extract_intent(text)
    if result.confidence < 0.6:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Low confidence task route. Ask for clarification or hand off to human.",
                "result": result.dict(),
            },
        )

    queue_name = DEPARTMENT_QUEUES.get(result.department, "front_desk_queue")
    priority = "high" if result.intent in {"maintenance_request", "room_service"} else "normal"

    return TaskRouteResult(
        intent=result.intent,
        department=result.department,
        queue=queue_name,
        priority=priority,
        confidence=result.confidence,
        needs_confirmation=result.needs_confirmation,
        should_create_task=True,
        raw_text=result.raw_text,
        source=result.source,
        route_reason=f"Routed to {queue_name} from {result.department}",
    )


@app.post("/create-task", response_model=TaskRecord)
async def create_task_route(payload: TaskCreateRequest, db: Session = Depends(get_db)):
    db_session, should_close = _resolve_db_session(db)
    routed = _route_text(payload.text)

    try:
        task = Task(
            text=payload.text,
            intent=routed.intent,
            department=routed.department,
            queue=routed.queue,
            priority=routed.priority,
            status="queued",
            confidence=routed.confidence,
            raw_text=routed.raw_text,
            source=routed.source,
            created_at=datetime.now(timezone.utc),
        )
        db_session.add(task)
        db_session.commit()
        db_session.refresh(task)

        return TaskRecord(
            id=task.id,
            text=task.text,
            intent=task.intent,
            department=task.department,
            queue=task.queue,
            priority=task.priority,
            status=task.status,
            confidence=task.confidence,
            raw_text=task.raw_text,
            source=task.source,
            created_at=task.created_at.isoformat(),
        )
    finally:
        if should_close:
            db_session.close()


@app.get("/tasks", response_model=List[TaskRecord])
async def list_tasks(db: Session = Depends(get_db)):
    db_session, should_close = _resolve_db_session(db)
    try:
        tasks = db_session.query(Task).order_by(Task.id.desc()).all()
        return [
            TaskRecord(
                id=task.id,
                text=task.text,
                intent=task.intent,
                department=task.department,
                queue=task.queue,
                priority=task.priority,
                status=task.status,
                confidence=task.confidence,
                raw_text=task.raw_text,
                source=task.source,
                created_at=task.created_at.isoformat(),
            )
            for task in tasks
        ]
    finally:
        if should_close:
            db_session.close()


@app.get("/tasks/department/{department}", response_model=List[TaskRecord])
async def list_tasks_by_department(department: AllowedDepartment, db: Session = Depends(get_db)):
    db_session, should_close = _resolve_db_session(db)
    try:
        tasks = db_session.query(Task).filter(Task.department == department).order_by(Task.id.desc()).all()
        return [
            TaskRecord(
                id=task.id,
                text=task.text,
                intent=task.intent,
                department=task.department,
                queue=task.queue,
                priority=task.priority,
                status=task.status,
                confidence=task.confidence,
                raw_text=task.raw_text,
                source=task.source,
                created_at=task.created_at.isoformat(),
            )
            for task in tasks
        ]
    finally:
        if should_close:
            db_session.close()


@app.post("/tasks/{task_id}/status/{new_status}", response_model=TaskRecord)
async def update_task_status(task_id: int, new_status: str, db: Session = Depends(get_db)):
    if new_status not in ("queued", "in_progress", "done"):
        raise HTTPException(status_code=400, detail="Invalid status. Must be 'queued', 'in_progress', or 'done'.")
    
    db_session, should_close = _resolve_db_session(db)
    try:
        task = db_session.query(Task).filter(Task.id == task_id).first()
        if not task:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found.")
        
        task.status = new_status
        db_session.commit()
        db_session.refresh(task)
        
        return TaskRecord(
            id=task.id,
            text=task.text,
            intent=task.intent,
            department=task.department,
            queue=task.queue,
            priority=task.priority,
            status=task.status,
            confidence=task.confidence,
            raw_text=task.raw_text,
            source=task.source,
            created_at=task.created_at.isoformat(),
        )
    finally:
        if should_close:
            db_session.close()

