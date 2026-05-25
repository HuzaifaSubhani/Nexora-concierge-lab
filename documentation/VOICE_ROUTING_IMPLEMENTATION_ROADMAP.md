# Voice Routing Implementation Roadmap

## Implementation Status (May 22, 2026)

- Completed: reusable `backend/app/routing/` package with contracts, config, model registry, detectors, orchestrator, transcription/translation/generation/persistence adapters.
- Completed: `backend/app/main.py` now delegates voice-routing decisions to orchestrator instead of inline endpoint logic.
- Completed: session mutation/persistence helpers in `session_manager.py` and endpoint flow now uses persistence-aware operations.
- Completed: audio probe + transcription profile routing path added through `services.py` and orchestrator transcription flow.
- Completed: duplicate model fallback logic removed by centralized model resolution.
- Completed: frontend voice router no longer auto-locks language from raw transcription hint; backend detection is the source of truth.
- Completed: additional backend tests for routing/model resolution and persistence helpers.

## Goal

Build a reusable routing layer that can accept either audio or text, detect the most likely language early, choose the right transcription and generation path, and expose the same logic to FastAPI, tests, background jobs, or another app without copying endpoint logic.

For this project, "routing layer" should mean:

1. Detect input modality: audio vs text.
2. Detect or estimate language as early as possible.
3. Select a transcription profile for audio inputs.
4. Select a translation strategy when English normalization is needed.
5. Select a language-specific LLM model and generation profile.
6. Return a structured routing decision that the API layer can use.

## What Already Exists

The current repo already has most of the building blocks:

- `backend/app/services.py`
  - Faster-Whisper transcription
  - Retry logic for poor transcripts
  - Best-effort translation into English
- `backend/app/language_router.py`
  - Script heuristics
  - `langdetect` fallback
  - Language confidence thresholding
  - DTMF fallback prompt generation
- `backend/app/llm_router.py`
  - Ollama request wrapper
  - Per-call model override support
  - Generation profile support
- `backend/app/session_manager.py`
  - Session lifecycle
  - State transitions
  - Persistence hooks
- `backend/app/main.py`
  - Current orchestration flow
  - Language-to-model map
  - Language-to-generation-profile map
- `frontend/app/voice-router/page.tsx`
  - Manual test harness for the voice routing flow

## Current Structure Summary

### Backend flow today

- `POST /transcribe`
  - audio -> Whisper transcript
- `POST /call/detect-language`
  - text -> language detection
- `POST /call/select-language`
  - DTMF fallback -> manual language lock
- `POST /call/voice-interact`
  - language hint/detection -> optional English translation -> LLM call -> locale formatting

### What is already good

- Language detection is separated enough to reuse.
- LLM invocation already supports model overrides.
- Transcription already returns a language hint.
- There is a session abstraction and a basic state machine.
- The frontend test page makes refactor verification easier.

## Gaps And Problems To Fix

### 1. The orchestration lives inside `main.py`

`backend/app/main.py` currently owns routing, translation, model selection, fallback handling, and session mutation. That makes the logic hard to reuse anywhere outside HTTP endpoints.

### 2. Persistence is inconsistent

The endpoints mutate `CallSession` objects directly instead of going through persistence-aware `SessionManager` operations. That means in-memory state and database state can drift until a later save.

Examples:

- `detect_language_route()` adds transcription and locks state directly.
- `select_language_route()` locks directly.
- `voice_interact_route()` adds transcription, adds response, and changes state directly.

### 3. Translation routing is conceptually mixed

`DEFAULT_TRANSLATION_MODEL_MAP` contains Hugging Face translation model IDs, but `voice_interact_route()` checks them against the list of Ollama models and then calls `llm_router.generate_response()` anyway. So the declared translation map and the actual translation execution path do not match.

### 4. Model fallback logic is duplicated

The main LLM fallback block is repeated twice inside `voice_interact_route()`. That should become one resolver function.

### 5. Output is not truly language-localized yet

The current flow normalizes user input into English before reasoning, but the response still comes back as English. `localization_engine.py` formats tokens like currency and date, but it does not translate the final response back into the user's language.

### 6. Frontend trusts Whisper language hints too early

`frontend/app/voice-router/page.tsx` marks the session as language-locked as soon as `/transcribe` returns any `language`, without backend confirmation or a separate confidence threshold for the language decision.

### 7. Audio and text are not routed through one contract

There is no unified "input routing" service yet. Audio takes one path and manual text takes another, with different assumptions and different lock behavior.

### 8. Old service copies create ambiguity

`services_old.py` and `services_backup.py` make it harder to tell which behavior is current and should not remain long term.

## What We Should Keep

These parts are worth keeping and extracting rather than rewriting:

- Script-based heuristics in `language_router.py`
- `_transcribe_with_options()` and Whisper setup in `services.py`
- `generate_response()` in `llm_router.py`
- Session state model in `session_manager.py`
- Model and generation maps in `main.py`, but moved into dedicated config/resolver modules

## Target Architecture

Create a reusable package under `backend/app/routing/` and keep FastAPI as a thin adapter.

### Proposed folder layout

```text
backend/app/routing/
  __init__.py
  contracts.py
  config.py
  model_registry.py
  decision_engine.py
  orchestrator.py
  detectors/
    __init__.py
    text_language_detector.py
    audio_language_probe.py
  transcription/
    __init__.py
    base.py
    whisper_engine.py
  translation/
    __init__.py
    base.py
    hf_translation_engine.py
    llm_translation_engine.py
  generation/
    __init__.py
    response_engine.py
  persistence/
    __init__.py
    session_repository.py
```

### Design rule

Everything in `routing/` should be pure application logic. It should not depend on FastAPI request objects or frontend assumptions.

### Core contracts

`contracts.py` should define small, explicit dataclasses or Pydantic models:

- `InputEnvelope`
  - `modality`: `"audio"` or `"text"`
  - `text`: optional
  - `audio_bytes`: optional
  - `session_id`: optional
  - `metadata`: dict
- `LanguageEvidence`
  - `language`
  - `confidence`
  - `source`
  - `lock_recommended`
- `TranscriptionResult`
  - `text`
  - `language`
  - `confidence`
  - `segments`
  - `engine`
- `TranslationResult`
  - `source_text`
  - `translated_text`
  - `source_language`
  - `target_language`
  - `engine`
- `ModelDecision`
  - `stt_profile`
  - `translation_engine`
  - `reasoning_model`
  - `generation_profile`
- `RouteResult`
  - full routing output used by API handlers

## Recommended Routing Pipeline

### For text input

1. Normalize input text.
2. Run text language detection.
3. Decide whether to lock language.
4. If non-English and the reasoning layer is English-first, translate to English.
5. Resolve LLM model and generation profile from language.
6. Generate response.
7. Translate response back to the user language if required by product behavior.

### For audio input

1. Run a lightweight audio language probe first.
2. Select a transcription profile based on probe output.
3. Run full transcription with that profile.
4. Re-check language from the final transcript.
5. Merge audio-language evidence and text-language evidence.
6. Decide whether to lock language.
7. Continue through translation and LLM routing.

### Why an audio probe first

Your stated goal is to redirect to a language-specific model for better transcription. To do that cleanly, we need a first-pass audio language estimate before the final transcription run.

The first implementation should keep this simple:

- use a fast multilingual probe
- resolve to a `TranscriptionProfile`
- then run the main transcription engine

This gives us language-aware transcription without hard-coding a giant `if/else` block inside the API endpoint.

## Transcription Profiles

Instead of binding routing directly to model names, introduce a `TranscriptionProfile`.

### Example

```json
{
  "default": {
    "engine": "faster_whisper",
    "model": "base",
    "language": null,
    "beam_size": 5,
    "vad_filter": true
  },
  "th": {
    "engine": "faster_whisper",
    "model": "base",
    "language": "th",
    "beam_size": 5,
    "vad_filter": false
  },
  "ar": {
    "engine": "faster_whisper",
    "model": "base",
    "language": "ar",
    "beam_size": 5,
    "vad_filter": true
  }
}
```

This lets us improve transcription per language even before we introduce separate STT models.

## Separation Of Concerns

### API layer should do only this

- validate request
- call routing orchestrator
- map result into response DTO

### Routing layer should do this

- detect language
- decide lock vs fallback
- choose STT/translation/LLM path
- return structured outputs

### Persistence layer should do this

- save session state
- save transcription history
- save responses
- hide DB details from route handlers

## Refactor Roadmap

### Phase 0: Stabilize What Exists

Goal: stop architectural drift before adding new behavior.

Tasks:

1. Remove duplicated fallback logic in `voice_interact_route()`.
2. Move model-map and generation-map resolution out of `main.py`.
3. Replace direct session mutation with repository or manager methods that persist immediately.
4. Decide which translation path is real:
   - Hugging Face / `translate_text()`
   - Ollama-based translation prompt
5. Mark old files as deprecated or remove them after validation.

Deliverable:

- existing behavior still works, but orchestration bugs and duplication are reduced

### Phase 1: Extract Routing Contracts

Goal: define reusable interfaces before more refactoring.

Tasks:

1. Add `routing/contracts.py`.
2. Add `routing/config.py` for language maps, generation maps, and transcription profiles.
3. Add `routing/model_registry.py` for model/profile lookup.
4. Add typed return objects for:
   - language detection
   - transcription
   - translation
   - model selection

Deliverable:

- endpoint code can pass around one structured route object instead of raw dicts and local variables

### Phase 2: Build A Unified Orchestrator

Goal: make one service handle both audio and text.

Tasks:

1. Create `routing/orchestrator.py`.
2. Expose methods like:
   - `route_text_input(...)`
   - `route_audio_input(...)`
   - `run_interaction(...)`
3. Move the logic now living in:
   - `/transcribe`
   - `/call/detect-language`
   - `/call/voice-interact`
   into the orchestrator layer.
4. Keep HTTP routes thin.

Deliverable:

- reusable routing service callable from API, CLI, tests, or workers

### Phase 3: Add Audio Language Probe + Transcription Profile Routing

Goal: support language-aware transcription cleanly.

Tasks:

1. Create `detectors/audio_language_probe.py`.
2. Probe early audio to estimate likely language.
3. Resolve a `TranscriptionProfile` from that estimate.
4. Run final transcription using the selected profile.
5. Reconcile probe result with transcript-based language detection.

Deliverable:

- audio no longer always follows a single generic transcription path

### Phase 4: Clean Translation Strategy

Goal: make language normalization explicit and consistent.

Tasks:

1. Decide whether the system is:
   - English-first internally, or
   - fully multilingual end to end
2. Implement translation engines behind one interface.
3. Keep "input translation" and "output translation" as separate steps.
4. Track which engine was used in the route result.

Deliverable:

- no more mixed translation semantics between Hugging Face IDs and Ollama model routing

### Phase 5: True Response Language Handling

Goal: return responses in the user's language, not only locale-formatted English.

Tasks:

1. Decide whether the LLM should answer directly in the user's language or answer in English and then translate.
2. Add an explicit response-language step.
3. Reserve `localization_engine.py` for formatting and regional behavior, not translation alone.
4. Make API response fields accurate:
   - `source_language`
   - `processing_language`
   - `response_language`

Deliverable:

- the response contract matches actual behavior

### Phase 6: Frontend Alignment

Goal: make UI behavior match backend truth.

Tasks:

1. Stop auto-locking language in the frontend purely because Whisper returned a `language`.
2. Let the backend return the authoritative lock decision.
3. Use one interaction endpoint response shape for both typed and spoken inputs.
4. Surface routing details for debugging:
   - detector used
   - transcription profile
   - translation engine
   - selected model

Deliverable:

- frontend becomes a thin client over the routing layer

### Phase 7: Packaging For Reuse

Goal: make the routing layer usable elsewhere.

Tasks:

1. Ensure `routing/` depends only on explicit interfaces.
2. Keep FastAPI-specific request/response DTOs outside the core package.
3. Add one service entrypoint like `VoiceRoutingService`.
4. Document how another caller can pass audio bytes or text and receive a `RouteResult`.

Deliverable:

- routing can be imported and reused in another project or service

## Suggested First Implementation Order

This is the safest order to actually code in follow-up sessions:

1. Extract config and resolver logic from `main.py`
2. Fix session persistence behavior
3. Create routing contracts
4. Create orchestrator for text flow first
5. Move audio flow into the orchestrator
6. Add audio probe + transcription profiles
7. Unify translation path
8. Add output-language support
9. Align frontend with backend routing decisions

## Tests We Should Add

### Unit tests

- text language detection with script heuristics
- language lock threshold decisions
- transcription profile resolution by language
- LLM model resolution by language
- translation engine resolution
- merged evidence decisions for audio probe + transcript

### Integration tests

- audio input -> probe -> transcription profile selected
- text input -> language detection -> model selection
- non-English input -> English normalization -> response generation
- fallback path when confidence is too low
- session persistence after detect, select, and interact steps

### Regression tests for current bugs

- session changes are persisted after `/call/detect-language`
- translation engine chosen is consistent with declared config
- `response_language` reflects actual response language
- duplicated fallback logic removed

## Acceptance Criteria

We can consider this routing layer "done" when:

1. The same orchestrator can process audio and text.
2. Language detection happens before final model selection.
3. Audio can select a language-aware transcription profile.
4. FastAPI endpoints are thin wrappers.
5. Session state is persisted consistently.
6. The returned response shape accurately describes:
   - input language
   - processing language
   - output language
   - transcription path
   - model path
7. The routing package can be imported by another module without FastAPI dependencies.

## Practical Notes For This Repo

### Best place for new modules

Use `backend/app/routing/` rather than adding more logic to `main.py`.

### Files likely to change first

- `backend/app/main.py`
- `backend/app/services.py`
- `backend/app/session_manager.py`
- `frontend/app/voice-router/page.tsx`

### Files likely to be created

- `backend/app/routing/contracts.py`
- `backend/app/routing/config.py`
- `backend/app/routing/model_registry.py`
- `backend/app/routing/orchestrator.py`
- `backend/app/routing/detectors/audio_language_probe.py`

## Recommendation

Do not start by adding more endpoints.

Start by extracting a pure routing service and moving the current decision logic there. Once that exists, we can improve transcription quality per language without increasing the coupling in `main.py`.

That gives us the modular foundation you asked for and makes every later improvement cheaper.
