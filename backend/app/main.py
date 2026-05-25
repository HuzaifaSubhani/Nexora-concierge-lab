import os
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

from fastapi import Depends, FastAPI, UploadFile, File, HTTPException
from fastapi.params import Depends as DependsParam
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.database import Base, SessionLocal, engine, get_db
from app.intents import extract_intent
from app.models import Task
from app.schemas import AllowedDepartment, IntentExtractionRequest, IntentExtractionResult, TaskCreateRequest, TaskRecord, TaskRouteRequest, TaskRouteResult
from app.services import warmup_whisper_model, translate_text
from app.schemas import TranslationRequest, TranslationResult
from app.llm_router import get_llm_router
from app.session_manager import get_session_manager, CallState
from app.routing import get_voice_routing_orchestrator

app = FastAPI(title="Nexora Backend - Milestone 3: Voice AI Receptionist with LLM Routing")
logger = logging.getLogger(__name__)

Base.metadata.create_all(bind=engine)

# Initialize routing components
llm_router = get_llm_router()
session_manager = get_session_manager()
voice_orchestrator = get_voice_routing_orchestrator()


def _prepare_intent_text(text: str) -> tuple[str, bool]:
    """Normalize request text for English-first intent extraction."""
    raw_text = (text or "").strip()
    if not raw_text:
        return "", False

    translated = translate_text(raw_text, "auto", "en").strip()
    if translated and translated.lower() != raw_text.lower():
        logger.info("Translated non-English intent input for extraction")
        return translated, True

    return raw_text, False


# ==================== Request/Response Models ====================

class InitiateCallRequest(BaseModel):
    """Request to initiate a new call session."""
    call_id: Optional[str] = None


class InitiateCallResponse(BaseModel):
    """Response with session ID and fallback prompt."""
    session_id: str
    call_id: str
    fallback_prompt: Optional[str] = None
    supported_languages: list


class DetectLanguageRequest(BaseModel):
    """Request to detect language from initial transcription."""
    session_id: str
    transcription: str


class DetectLanguageResponse(BaseModel):
    """Response with detected language and confidence."""
    session_id: str
    detected_language: str
    confidence: float
    language_locked: bool
    confirmation_required: bool = True
    confirmation_prompt: Optional[str] = None


class ConfirmLanguageRequest(BaseModel):
    """Request to confirm or reject auto-detected language."""
    session_id: str
    confirm: bool
    selected_language: Optional[str] = None


class ConfirmLanguageResponse(BaseModel):
    """Response after language confirmation/rejection."""
    session_id: str
    selected_language: str
    language_locked: bool
    message: str
    voice_prompt: Optional[str] = None


class SelectLanguageRequest(BaseModel):
    """Request to manually select language via DTMF."""
    session_id: str
    dtmf_key: str


class SelectLanguageResponse(BaseModel):
    """Response confirming language selection."""
    session_id: str
    selected_language: str
    language_locked: bool


class VoiceInteractionRequest(BaseModel):
    """Request for LLM to process transcribed voice input."""
    session_id: str
    transcription: str
    source_language: Optional[str] = None
    stream: bool = False


class VoiceInteractionResponse(BaseModel):
    """Response with LLM-generated reply."""
    session_id: str
    transcription: str
    translated_transcription: str
    response: str
    language: str
    response_language: str
    source_language: str
    requested_model: str
    selected_model: str
    model_fallback_used: bool
    applied_generation_profile: Dict[str, object]
    translation_model: str
    translation_used: bool
    latency_ms: float


class EndCallRequest(BaseModel):
    """Request to end call session."""
    session_id: str
    reason: str = "completed"


# ==================== Health & Status ====================


@app.on_event("startup")
async def preload_whisper_model_on_startup():
    preload = str(os.getenv("NEXORA_WHISPER_PRELOAD", "false")).strip().lower() in {"1", "true", "yes", "on"}
    if not preload:
        return

    try:
        info = warmup_whisper_model()
        print(f"[INFO] Whisper preload complete: {info}")
    except Exception as exc:
        # Keep API startup resilient; model can still lazy-load on first request.
        print(f"[WARN] Whisper preload failed: {exc}")
    
    # Check LLM router health
    try:
        llm_healthy = await llm_router.health_check()
        if llm_healthy:
            print(f"[INFO] LLM (Ollama) health check passed")
        else:
            print(f"[WARN] LLM (Ollama) not fully ready; will attempt lazy-load on first request")
    except Exception as e:
        print(f"[WARN] LLM health check failed: {e}")

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
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/ping")
async def ping():
    return {"status": "ok", "message": "pong"}


@app.get("/")
async def root():
    return {
        "message": "Nexora backend is running",
        "docs": "/docs",
        "health": "/health",
        "ping": "/ping",
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/call/model-routing")
async def model_routing_status():
    """Return effective language->model mapping plus installed Ollama models."""
    return await voice_orchestrator.get_routing_status()


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """
    Transcribe audio file to text.
    
    Accepts: wav, mp3, m4a, flac, etc.
    Returns: {"text": "...", "confidence": 0.95, "language": "en", "status": "success"}
    """
    audio_data = await file.read()
    result = await voice_orchestrator.transcribe_audio_input(audio_data)
    return {
        "text": result.text,
        "confidence": result.confidence,
        "language": result.language,
        "status": result.status,
        "segments": result.segments,
        "engine": result.engine,
        "translation": result.translation,
        "translation_language": result.translation_language,
        "transcription_profile": (
            {
                "model": result.profile.model,
                "language": result.profile.language,
                "beam_size": result.profile.beam_size,
                "vad_filter": result.profile.vad_filter,
            }
            if result.profile
            else None
        ),
    }


@app.post("/transcribe/warmup")
async def warmup_transcribe_model():
    """Preload faster-whisper model to avoid first-request timeout spikes."""
    return warmup_whisper_model()


@app.post("/translate", response_model=TranslationResult)
async def translate_route(payload: TranslationRequest):
    """Translate supplied text (auto-detects language when src_lang is omitted)."""
    src = payload.src_lang or "auto"
    translation = translate_text(payload.text, src, "en")
    if not translation:
        raise HTTPException(status_code=500, detail="Translation unavailable")
    return TranslationResult(translation=translation, translation_language="en")


@app.post("/extract-intent", response_model=IntentExtractionResult)
async def extract_intent_route(payload: IntentExtractionRequest):
    intent_text, translated = _prepare_intent_text(payload.text)
    result = extract_intent(intent_text)
    if translated:
        result = result.copy(update={"raw_text": payload.text})
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
    intent_text, translated = _prepare_intent_text(text)
    result = extract_intent(intent_text)
    if translated:
        result = result.copy(update={"raw_text": text})
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


# ==================== Voice Routing Endpoints ====================


@app.post("/call/initiate", response_model=InitiateCallResponse)
async def initiate_call(payload: InitiateCallRequest):
    """Create a new call session and return fallback prompt."""
    session = session_manager.create_session(call_id=payload.call_id)
    session_manager.update_session_state(session.session_id, CallState.LANGUAGE_DETECTING)
    fallback_prompt = voice_orchestrator.get_fallback_prompt("en")

    return InitiateCallResponse(
        session_id=session.session_id,
        call_id=session.call_id,
        fallback_prompt=fallback_prompt,
        supported_languages=voice_orchestrator.get_supported_languages(),
    )


@app.post("/call/detect-language", response_model=DetectLanguageResponse)
async def detect_language_route(payload: DetectLanguageRequest):
    """Detect language from initial transcription and optionally lock language."""
    if not session_manager.get_session(payload.session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    outcome = voice_orchestrator.detect_language_for_session(payload.session_id, payload.transcription)

    return DetectLanguageResponse(
        session_id=payload.session_id,
        detected_language=outcome["detected_language"],
        confidence=outcome["confidence"],
        language_locked=outcome["language_locked"],
        confirmation_required=outcome["confirmation_required"],
        confirmation_prompt=outcome["confirmation_prompt"],
    )


@app.post("/call/confirm-language", response_model=ConfirmLanguageResponse)
async def confirm_language_route(payload: ConfirmLanguageRequest):
    """Confirm or reject detected language before locking session language."""
    if not session_manager.get_session(payload.session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    outcome = voice_orchestrator.confirm_language_for_session(
        session_id=payload.session_id,
        confirm=payload.confirm,
        selected_language=payload.selected_language,
    )
    return ConfirmLanguageResponse(
        session_id=outcome["session_id"],
        selected_language=outcome["selected_language"],
        language_locked=outcome["language_locked"],
        message=outcome["message"],
        voice_prompt=outcome.get("voice_prompt"),
    )


@app.post("/call/select-language", response_model=SelectLanguageResponse)
async def select_language_route(payload: SelectLanguageRequest):
    """Manually select language via DTMF selection."""
    if not session_manager.get_session(payload.session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    try:
        selected = voice_orchestrator.select_language_for_session(payload.session_id, payload.dtmf_key)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid DTMF selection")

    return SelectLanguageResponse(session_id=payload.session_id, selected_language=selected, language_locked=True)


@app.post("/call/voice-interact", response_model=VoiceInteractionResponse)
async def voice_interact_route(payload: VoiceInteractionRequest):
    """Process a transcribed voice segment through the LLM and return localized response."""
    if not session_manager.get_session(payload.session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    try:
        routed = await voice_orchestrator.run_voice_interaction(
            session_id=payload.session_id,
            transcription=payload.transcription,
            source_language_hint=payload.source_language,
            stream=payload.stream,
        )
    except ValueError as exc:
        detail = str(exc)
        if "not confirmed" in detail.lower():
            raise HTTPException(status_code=409, detail=detail)
        raise HTTPException(status_code=404, detail=detail)

    return VoiceInteractionResponse(
        session_id=routed.session_id,
        transcription=routed.transcription,
        translated_transcription=routed.translated_transcription,
        response=routed.response_text,
        language=routed.processing_language,
        response_language=routed.response_language,
        source_language=routed.source_language,
        requested_model=routed.requested_model,
        selected_model=routed.selected_model,
        model_fallback_used=routed.model_fallback_used,
        applied_generation_profile=routed.applied_generation_profile,
        translation_model=routed.translation_model,
        translation_used=routed.translation_used,
        latency_ms=routed.latency_ms,
    )


@app.post("/call/end", status_code=204)
async def end_call_route(payload: EndCallRequest):
    session = session_manager.get_session(payload.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    session_manager.close_session(payload.session_id, payload.reason)
    return {}


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

