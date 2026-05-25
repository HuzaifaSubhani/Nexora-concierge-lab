"""Core contracts for reusable voice routing."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class InputEnvelope:
    """Normalized input contract for text or audio workflows."""

    modality: str
    session_id: Optional[str] = None
    text: Optional[str] = None
    audio_bytes: Optional[bytes] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LanguageEvidence:
    """Language decision details with provenance."""

    language: str
    confidence: float
    source: str
    lock_recommended: bool


@dataclass
class TranscriptionProfile:
    """Profile used for language-aware STT routing."""

    model: str
    language: Optional[str]
    beam_size: int
    vad_filter: bool


@dataclass
class TranscriptionResult:
    """Normalized transcription output."""

    text: str
    confidence: float
    language: str
    status: str
    segments: int = 0
    engine: str = "faster-whisper"
    translation: Optional[str] = None
    translation_language: Optional[str] = None
    profile: Optional[TranscriptionProfile] = None


@dataclass
class ModelDecision:
    """LLM model selection details."""

    requested_model: str
    selected_model: str
    model_fallback_used: bool
    generation_profile: Dict[str, Any]


@dataclass
class InteractionRouteResult:
    """Result returned by the orchestration layer for voice interaction."""

    session_id: str
    source_language: str
    source_language_confidence: float
    language_locked: bool
    transcription: str
    translated_transcription: str
    processing_language: str
    response_text: str
    response_language: str
    translation_used: bool
    translation_model: str
    requested_model: str
    selected_model: str
    model_fallback_used: bool
    applied_generation_profile: Dict[str, Any]
    latency_ms: float
