"""Base contract for transcription engines."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from app.routing.contracts import TranscriptionResult


class TranscriptionEngine(ABC):
    """Abstract transcription engine contract."""

    @abstractmethod
    async def transcribe(
        self,
        audio_data: bytes,
        profile: Optional[Dict[str, Any]] = None,
        probe_result: Optional[Dict[str, Any]] = None,
    ) -> TranscriptionResult:
        raise NotImplementedError
