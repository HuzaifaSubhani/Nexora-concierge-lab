"""Reusable orchestration layer for voice and language routing."""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

from app.language_router import get_language_router
from app.llm_router import get_llm_router
from app.localization_engine import get_localization_engine
from app.routing.config import get_effective_generation_map, get_effective_model_map
from app.routing.contracts import (
    InteractionRouteResult,
    LanguageEvidence,
    TranscriptionProfile,
    TranscriptionResult,
)
from app.routing.detectors import AudioLanguageProbe
from app.routing.model_registry import (
    resolve_transcription_profile_for_language,
    resolve_translation_model_for_language,
)
from app.routing.persistence import SessionRepository
from app.routing.transcription import WhisperTranscriptionEngine
from app.routing.translation import HFTranslationEngine
from app.receptionist import (
    ASK_LANGUAGE_PROMPT,
    INITIAL_GREETING,
    LANGUAGE_DETECTION_FAILED_PROMPT,
    LANGUAGE_NAMES,
    SUPPORTED_LANGUAGE_CODES,
    TTS_LANGUAGES,
    UNSUPPORTED_LANGUAGE_PROMPT,
    confirmation_prompt_for_language,
    detect_supported_language,
    english_meaning_from_keywords,
    language_code_from_name_or_code,
    query_prompt_for_language,
    route_receptionist_query,
)
from app.services import normalize_repeated_text
from app.session_manager import CallState, get_session_manager


class VoiceRoutingOrchestrator:
    """Single entrypoint for reusable language-aware voice orchestration."""

    def __init__(self):
        self.language_router = get_language_router()
        self.llm_router = get_llm_router()
        self.localization_engine = get_localization_engine()
        self.session_manager = get_session_manager()
        self.session_repo = SessionRepository(self.session_manager)
        self.audio_probe = AudioLanguageProbe()
        self.transcription_engine = WhisperTranscriptionEngine()
        self.translation_engine = HFTranslationEngine()

    async def get_routing_status(self) -> Dict[str, Any]:
        return {
            "default_model": self.llm_router.model,
            "effective_map": get_effective_model_map(),
            "generation_map": get_effective_generation_map(),
            "available_models": await self.llm_router.list_available_models(),
        }

    def get_supported_languages(self) -> list[str]:
        return list(SUPPORTED_LANGUAGE_CODES)

    def get_fallback_prompt(self, detected_lang: str = "en") -> str:
        return INITIAL_GREETING

    def _build_language_confirmation_prompt(self, language: str, transcription: str) -> str:
        return confirmation_prompt_for_language(language)

    def detect_language_for_session(self, session_id: str, transcription: str) -> Dict[str, Any]:
        session = self.session_manager.get_session(session_id)
        if not session:
            raise ValueError("Session not found")

        detection = detect_supported_language(transcription)
        self.session_repo.add_transcription(session_id, transcription)
        self.session_repo.set_confidence(session_id, detection.confidence)
        self.session_repo.transition(session_id, CallState.LANGUAGE_DETECTING)
        self.session_repo.set_metadata_value(session_id, "pending_language", detection.code)
        self.session_repo.set_metadata_value(session_id, "pending_language_confidence", detection.confidence)
        self.session_repo.set_metadata_value(session_id, "pending_transcription", transcription)

        if detection.code == "unknown":
            confirmation_prompt = LANGUAGE_DETECTION_FAILED_PROMPT
            self.session_repo.set_metadata_value(session_id, "pending_confirmation_prompt", confirmation_prompt)
            return {
                "state": "LANGUAGE_DETECTION",
                "detected_language": "unknown",
                "detected_language_name": "Unknown",
                "confirmed_language": None,
                "language_confirmed": False,
                "confidence": 0.0,
                "language_locked": False,
                "confirmation_required": True,
                "confirmation_prompt": confirmation_prompt,
                "caller_text_original": transcription,
                "caller_text_english": transcription,
                "intent": "Unknown",
                "response_english": confirmation_prompt,
                "response_final_language": confirmation_prompt,
                "action": "ask_clarification",
                "tts_language": "en",
                "notes": "Language detection failed.",
            }

        confirmation_prompt = self._build_language_confirmation_prompt(detection.code, transcription)
        self.session_repo.set_metadata_value(session_id, "pending_confirmation_prompt", confirmation_prompt)

        return {
            "state": "LANGUAGE_CONFIRMATION",
            "detected_language": detection.code,
            "detected_language_name": detection.name,
            "confirmed_language": None,
            "language_confirmed": False,
            "confidence": detection.confidence,
            "language_locked": False,
            "confirmation_required": True,
            "confirmation_prompt": confirmation_prompt,
            "caller_text_original": transcription,
            "caller_text_english": transcription,
            "intent": "Unknown",
            "response_english": confirmation_prompt,
            "response_final_language": confirmation_prompt,
            "action": "ask_language_confirmation",
            "tts_language": TTS_LANGUAGES.get(detection.code, "en"),
            "notes": "Detected language requires caller confirmation.",
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
        selected_code = language_code_from_name_or_code(selected_language)
        locked_language = selected_code or pending_language

        if confirm:
            if locked_language not in SUPPORTED_LANGUAGE_CODES:
                return {
                    "state": "LANGUAGE_CONFIRMATION",
                    "session_id": session_id,
                    "selected_language": "unknown",
                    "detected_language": "unknown",
                    "confirmed_language": None,
                    "language_confirmed": False,
                    "language_locked": False,
                    "message": "Unsupported language.",
                    "voice_prompt": UNSUPPORTED_LANGUAGE_PROMPT,
                    "response_english": UNSUPPORTED_LANGUAGE_PROMPT,
                    "response_final_language": UNSUPPORTED_LANGUAGE_PROMPT,
                    "action": "ask_language_confirmation",
                    "tts_language": "en",
                    "intent": "Unknown",
                    "notes": "Caller selected an unsupported language.",
                }
            if selected_code:
                pending_confidence = max(90.0, pending_confidence)
            self.session_repo.lock_language(session_id, locked_language, pending_confidence or 90.0)
            self.session_repo.set_metadata_value(session_id, "pending_language", None)
            self.session_repo.set_metadata_value(session_id, "pending_language_confidence", None)
            self.session_repo.set_metadata_value(session_id, "pending_transcription", None)
            self.session_repo.set_metadata_value(session_id, "pending_confirmation_prompt", None)
            self.session_repo.set_metadata_value(session_id, "routing_failure_count", 0)
            self.session_repo.set_metadata_value(session_id, "no_speech_count", 0)
            voice_prompt = query_prompt_for_language(locked_language)
            return {
                "state": "LISTENING_TO_QUERY",
                "session_id": session_id,
                "selected_language": locked_language,
                "detected_language": locked_language,
                "confirmed_language": LANGUAGE_NAMES[locked_language],
                "language_confirmed": True,
                "language_locked": True,
                "message": f"Confirmed. Continuing in {LANGUAGE_NAMES[locked_language]}.",
                "voice_prompt": voice_prompt,
                "response_english": query_prompt_for_language("en"),
                "response_final_language": voice_prompt,
                "action": "ask_query",
                "tts_language": TTS_LANGUAGES.get(locked_language, "en"),
                "intent": "Unknown",
                "notes": "Language locked for the call.",
            }

        # User rejected detected language.
        self.session_repo.transition(session_id, CallState.LANGUAGE_DETECTING)
        return {
            "state": "LANGUAGE_CONFIRMATION",
            "session_id": session_id,
            "selected_language": pending_language or "en",
            "detected_language": pending_language or "unknown",
            "confirmed_language": None,
            "language_confirmed": False,
            "language_locked": False,
            "message": "Language not confirmed. Please choose your preferred language.",
            "voice_prompt": ASK_LANGUAGE_PROMPT,
            "response_english": ASK_LANGUAGE_PROMPT,
            "response_final_language": ASK_LANGUAGE_PROMPT,
            "action": "ask_language_confirmation",
            "tts_language": "en",
            "intent": "Unknown",
            "notes": "Caller rejected detected language.",
        }

    def select_language_for_session(self, session_id: str, dtmf_key: str) -> str:
        session = self.session_manager.get_session(session_id)
        if not session:
            raise ValueError("Session not found")

        dtmf_map = {
            "1": "en",
            "2": "fr",
            "3": "it",
            "4": "de",
            "5": "nl",
            "6": "es",
        }
        selected = dtmf_map.get(dtmf_key.strip()) or language_code_from_name_or_code(dtmf_key)
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
        start_time = time.time()
        session = self.session_manager.get_session(session_id)
        if not session:
            raise ValueError("Session not found")

        transcription = normalize_repeated_text(transcription)
        evidence = self._get_interaction_language(session_id, transcription, source_language_hint)
        self.session_repo.add_transcription(session_id, transcription)
        self.session_repo.transition(session_id, CallState.ROUTING)

        detected_lang = evidence.language or "en"
        translated_transcription = transcription
        translation_used = False
        translation_model = resolve_translation_model_for_language(detected_lang)

        # Normalize user text into English before main reasoning.
        if detected_lang != "en":
            keyword_meaning = english_meaning_from_keywords(transcription, detected_lang)
            if keyword_meaning and keyword_meaning != transcription:
                translated_transcription = keyword_meaning
                translation_used = True
                translation_model = "rule_based_meaning"
            elif os.getenv("NEXORA_ENABLE_TRANSLATION_ENGINE", "").strip().lower() in {"1", "true", "yes", "on"}:
                translated = self.translation_engine.translate(transcription.strip(), detected_lang, "en")
                cleaned_translation = normalize_repeated_text(translated)
                if cleaned_translation and cleaned_translation.lower() != transcription.strip().lower():
                    translated_transcription = cleaned_translation
                    translation_used = True
                else:
                    translation_model = "translation_unavailable"
            else:
                translation_model = "rule_based_meaning"
        else:
            translation_model = "direct"

        if detected_lang != "en" and (not translated_transcription or translated_transcription == transcription):
            translated_transcription = transcription

        session = self.session_manager.get_session(session_id)
        failure_count = int((session.metadata if session else {}).get("routing_failure_count", 0) or 0)
        decision = route_receptionist_query(
            caller_text_original=transcription,
            caller_text_english=translated_transcription,
            language_code=detected_lang,
            failure_count=failure_count,
        )

        if decision.intent == "Unknown" and decision.action == "ask_clarification":
            self.session_repo.set_metadata_value(session_id, "routing_failure_count", failure_count + 1)
        else:
            self.session_repo.set_metadata_value(session_id, "routing_failure_count", 0)

        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000.0

        response_language = detected_lang if detected_lang != "en" else "en"
        localized = self.localization_engine.inject_locale_context(decision.response_final_language, detected_lang)
        self.session_repo.add_response(session_id, localized, source="receptionist_rules", latency_ms=latency_ms)
        if decision.action == "end_call":
            self.session_repo.close(session_id, "caller_ended")
            result_state = "END_CALL"
        elif decision.action in {"route_department", "transfer_human"}:
            self.session_repo.transition(session_id, CallState.IN_CONVERSATION)
            result_state = "CONTINUE_OR_TRANSFER"
        else:
            self.session_repo.transition(session_id, CallState.IN_CONVERSATION)
            result_state = "LISTENING_TO_QUERY"

        session = self.session_manager.get_session(session_id)
        return InteractionRouteResult(
            session_id=session_id,
            state=result_state,
            detected_language_name=LANGUAGE_NAMES.get(detected_lang, "Unknown"),
            confirmed_language_name=LANGUAGE_NAMES.get(detected_lang),
            language_confirmed=bool(session.language_locked if session else evidence.lock_recommended),
            source_language=detected_lang,
            source_language_confidence=float(session.language_confidence if session else evidence.confidence),
            language_locked=bool(session.language_locked if session else evidence.lock_recommended),
            transcription=transcription,
            caller_text_original=transcription,
            caller_text_english=translated_transcription,
            translated_transcription=translated_transcription,
            processing_language="en",
            response_text=localized,
            response_language=response_language,
            response_english=decision.response_english,
            response_final_language=localized,
            intent=decision.intent,
            confidence=decision.confidence,
            action=decision.action,
            tts_language=TTS_LANGUAGES.get(detected_lang, "en"),
            notes=decision.notes,
            translation_used=translation_used,
            translation_model=translation_model,
            requested_model="rule_based_receptionist",
            selected_model="rule_based_receptionist",
            model_fallback_used=False,
            applied_generation_profile={"max_spoken_words": 35, "template": True},
            latency_ms=latency_ms,
        )


_orchestrator_instance: Optional[VoiceRoutingOrchestrator] = None


def get_voice_routing_orchestrator() -> VoiceRoutingOrchestrator:
    """Get global orchestrator instance."""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = VoiceRoutingOrchestrator()
    return _orchestrator_instance
