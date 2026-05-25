"""Reusable orchestration layer for voice and language routing."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from app.language_router import SUPPORTED_LANGUAGES, get_language_router
from app.llm_router import get_llm_router
from app.localization_engine import get_localization_engine
from app.prompts import get_system_prompt
from app.routing.config import get_effective_generation_map, get_effective_model_map, get_supported_languages
from app.routing.contracts import (
    InteractionRouteResult,
    LanguageEvidence,
    TranscriptionProfile,
    TranscriptionResult,
)
from app.routing.detectors import AudioLanguageProbe, TextLanguageDetector
from app.routing.generation import ResponseGenerationEngine
from app.routing.model_registry import (
    resolve_model_for_language,
    resolve_transcription_profile_for_language,
    resolve_translation_model_for_language,
)
from app.routing.persistence import SessionRepository
from app.routing.transcription import WhisperTranscriptionEngine
from app.routing.translation import HFTranslationEngine
from app.session_manager import CallState, get_session_manager


class VoiceRoutingOrchestrator:
    """Single entrypoint for reusable language-aware voice orchestration."""

    def __init__(self):
        self.language_router = get_language_router()
        self.llm_router = get_llm_router()
        self.localization_engine = get_localization_engine()
        self.session_manager = get_session_manager()
        self.session_repo = SessionRepository(self.session_manager)
        self.text_detector = TextLanguageDetector(self.language_router)
        self.audio_probe = AudioLanguageProbe()
        self.transcription_engine = WhisperTranscriptionEngine()
        self.translation_engine = HFTranslationEngine()
        self.response_engine = ResponseGenerationEngine(self.llm_router)

    async def get_routing_status(self) -> Dict[str, Any]:
        return {
            "default_model": self.llm_router.model,
            "effective_map": get_effective_model_map(),
            "generation_map": get_effective_generation_map(),
            "available_models": await self.llm_router.list_available_models(),
        }

    def get_supported_languages(self) -> list[str]:
        return get_supported_languages()

    def get_fallback_prompt(self, detected_lang: str = "en") -> str:
        prompt, _ = self.language_router.get_fallback_prompt(detected_lang)
        return prompt

    def _build_language_confirmation_prompt(self, language: str, transcription: str) -> str:
        language_name = SUPPORTED_LANGUAGES.get(language, language.upper())
        snippet = (transcription or "").strip()
        if len(snippet) > 160:
            snippet = snippet[:157].rstrip() + "..."
        return (
            f'I heard: "{snippet}". '
            f"Should I continue in {language_name}? "
            "Say yes to continue, or say no and choose another language."
        )

    def detect_language_for_session(self, session_id: str, transcription: str) -> Dict[str, Any]:
        session = self.session_manager.get_session(session_id)
        if not session:
            raise ValueError("Session not found")

        evidence = self.text_detector.detect(transcription)
        self.session_repo.add_transcription(session_id, transcription)
        self.session_repo.set_confidence(session_id, evidence.confidence)
        self.session_repo.transition(session_id, CallState.LANGUAGE_DETECTING)
        self.session_repo.set_metadata_value(session_id, "pending_language", evidence.language)
        self.session_repo.set_metadata_value(session_id, "pending_language_confidence", evidence.confidence)
        self.session_repo.set_metadata_value(session_id, "pending_transcription", transcription)

        confirmation_prompt = self._build_language_confirmation_prompt(evidence.language, transcription)
        self.session_repo.set_metadata_value(session_id, "pending_confirmation_prompt", confirmation_prompt)

        return {
            "detected_language": evidence.language,
            "confidence": evidence.confidence,
            "language_locked": False,
            "confirmation_required": True,
            "confirmation_prompt": confirmation_prompt,
        }

    def confirm_language_for_session(
        self,
        session_id: str,
        confirm: bool,
        selected_language: Optional[str] = None,
    ) -> Dict[str, Any]:
        session = self.session_manager.get_session(session_id)
        if not session:
            raise ValueError("Session not found")

        pending_language = str(session.metadata.get("pending_language", "") or "").strip().lower()
        pending_confidence = float(session.metadata.get("pending_language_confidence", 0.0) or 0.0)
        locked_language = pending_language or "en"

        if confirm:
            if selected_language and selected_language.strip():
                locked_language = selected_language.strip().lower()
                pending_confidence = max(90.0, pending_confidence)
            self.session_repo.lock_language(session_id, locked_language, pending_confidence or 90.0)
            self.session_repo.set_metadata_value(session_id, "pending_language", None)
            self.session_repo.set_metadata_value(session_id, "pending_language_confidence", None)
            self.session_repo.set_metadata_value(session_id, "pending_transcription", None)
            self.session_repo.set_metadata_value(session_id, "pending_confirmation_prompt", None)
            return {
                "session_id": session_id,
                "selected_language": locked_language,
                "language_locked": True,
                "message": f"Confirmed. Continuing in {locked_language.upper()}.",
                "voice_prompt": f"Great. We will continue in {locked_language.upper()}.",
            }

        # User rejected detected language.
        self.session_repo.transition(session_id, CallState.LANGUAGE_DETECTING)
        fallback_prompt = self.get_fallback_prompt(pending_language or "en")
        return {
            "session_id": session_id,
            "selected_language": pending_language or "en",
            "language_locked": False,
            "message": "Language not confirmed. Please choose your preferred language.",
            "voice_prompt": fallback_prompt,
        }

    def select_language_for_session(self, session_id: str, dtmf_key: str) -> str:
        session = self.session_manager.get_session(session_id)
        if not session:
            raise ValueError("Session not found")

        selected = self.language_router.route_by_dtmf_selection(dtmf_key)
        if not selected:
            raise ValueError("Invalid DTMF selection")

        self.session_repo.lock_language(session_id, selected, 100.0)
        return selected

    async def transcribe_audio_input(self, audio_data: bytes) -> TranscriptionResult:
        probe = self.audio_probe.probe(audio_data)
        probe_language = probe.language if probe.confidence >= 55.0 else None
        profile = resolve_transcription_profile_for_language(probe_language)

        probe_result = {
            "language": probe.language,
            "confidence": round(probe.confidence / 100.0, 3),
            "status": "success",
            "source": probe.source,
        }

        result = await self.transcription_engine.transcribe(
            audio_data=audio_data,
            profile={
                "model": profile.model,
                "language": profile.language,
                "beam_size": profile.beam_size,
                "vad_filter": profile.vad_filter,
            },
            probe_result=probe_result,
        )
        if result.profile is None:
            result.profile = TranscriptionProfile(
                model=profile.model,
                language=profile.language,
                beam_size=profile.beam_size,
                vad_filter=profile.vad_filter,
            )
        return result

    def _get_interaction_language(self, session_id: str, transcription: str, source_language_hint: Optional[str]) -> LanguageEvidence:
        session = self.session_manager.get_session(session_id)
        if not session:
            raise ValueError("Session not found")

        if not session.language_locked:
            raise ValueError("Language not confirmed. Please confirm language before continuing.")

        return LanguageEvidence(
            language=session.language,
            confidence=session.language_confidence,
            source="session_lock",
            lock_recommended=True,
        )

    async def run_voice_interaction(
        self,
        session_id: str,
        transcription: str,
        source_language_hint: Optional[str] = None,
        stream: bool = False,
    ) -> InteractionRouteResult:
        session = self.session_manager.get_session(session_id)
        if not session:
            raise ValueError("Session not found")

        transcription = normalize_repeated_text(transcription)
        evidence = self._get_interaction_language(session_id, transcription, source_language_hint)
        self.session_repo.add_transcription(session_id, transcription)

        detected_lang = evidence.language or "en"
        translated_transcription = transcription
        translation_used = False
        translation_model = resolve_translation_model_for_language(detected_lang)

        # Normalize user text into English before main reasoning.
        if detected_lang != "en":
            translated = self.translation_engine.translate(transcription.strip(), detected_lang, "en")
            if translated:
                translated_transcription = normalize_repeated_text(translated)
                translation_used = True

        available_models = await self.llm_router.list_available_models()
        model_decision = resolve_model_for_language(
            language=detected_lang,
            default_model=self.llm_router.model,
            available_models=available_models,
        )

        system_prompt = get_system_prompt("en")
        start_time = time.time()
        llm_response = await self.response_engine.generate(
            text=translated_transcription,
            language="en",
            system_prompt=system_prompt,
            model_override=model_decision.selected_model,
            generation_profile=model_decision.generation_profile,
            stream=stream,
        )
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000.0

        response_en = ""
        if hasattr(llm_response, "__aiter__"):
            async for chunk in llm_response:
                response_en += chunk
        else:
            response_en = llm_response or ""

        response_text = response_en
        response_language = "en"
        if detected_lang != "en":
            translated_out = self.translation_engine.translate(response_en.strip(), "en", detected_lang)
            if translated_out:
                response_text = translated_out
                response_language = detected_lang

        localized = self.localization_engine.inject_locale_context(response_text, detected_lang)
        self.session_repo.add_response(session_id, localized, source="llm", latency_ms=latency_ms)
        self.session_repo.transition(session_id, CallState.IN_CONVERSATION)

        session = self.session_manager.get_session(session_id)
        return InteractionRouteResult(
            session_id=session_id,
            source_language=detected_lang,
            source_language_confidence=float(session.language_confidence if session else evidence.confidence),
            language_locked=bool(session.language_locked if session else evidence.lock_recommended),
            transcription=transcription,
            translated_transcription=translated_transcription,
            processing_language="en",
            response_text=localized,
            response_language=response_language,
            translation_used=translation_used,
            translation_model=translation_model,
            requested_model=model_decision.requested_model,
            selected_model=model_decision.selected_model,
            model_fallback_used=model_decision.model_fallback_used,
            applied_generation_profile=model_decision.generation_profile,
            latency_ms=latency_ms,
        )


_orchestrator_instance: Optional[VoiceRoutingOrchestrator] = None


def get_voice_routing_orchestrator() -> VoiceRoutingOrchestrator:
    """Get global orchestrator instance."""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = VoiceRoutingOrchestrator()
    return _orchestrator_instance
