"""
Transcription and AI services for Nexora.
"""
import os
import re
import unicodedata
import tempfile
from typing import Any, Dict, Optional

WhisperModel = None
_whisper_import_error = None
TranslationPipeline = None
_translator_cache = {}
_google_translator = None
_PHRASE_TRANSLATIONS = {
    "fr": {
        "j ai besoin de serviettes propres": "I need fresh towels",
        "la climatisation ne fonctionne pas": "The AC is not working",
        "veuillez envoyer le service en chambre": "Please send room service",
        "l ampoule est cassee": "The light bulb is broken",
        "je souhaite un depart tardif": "I need a late checkout",
        "ma commande de diner manque": "My dinner order is missing",
    },
    "de": {
        "ich brauche frische handtucher": "I need fresh towels",
        "die klimaanlage funktioniert nicht": "The AC is not working",
        "bitte schicken sie den zimmerservice": "Please send room service",
        "die gluhbirne ist kaputt": "The light bulb is broken",
        "ich brauche einen spaten check out": "I need a late checkout",
        "meine essensbestellung fehlt": "My dinner order is missing",
    },
    "it": {
        "ho bisogno di asciugamani puliti": "I need fresh towels",
        "l aria condizionata non funziona": "The AC is not working",
        "per favore inviate il servizio in camera": "Please send room service",
        "la lampadina e rotta": "The light bulb is broken",
        "ho bisogno di un check out tardivo": "I need a late checkout",
        "manca il mio ordine di cena": "My dinner order is missing",
    },
    "es": {
        "necesito toallas limpias": "I need fresh towels",
        "el aire acondicionado no funciona": "The AC is not working",
        "por favor envien servicio a la habitacion": "Please send room service",
        "la bombilla esta rota": "The light bulb is broken",
        "necesito un check out tardio": "I need a late checkout",
        "falta mi pedido de cena": "My dinner order is missing",
    },
}

_PHRASE_TRANSLATION_RULES = {
    "fr": [
        ("serviette", "I need fresh towels"),
        ("climatisation", "The AC is not working"),
        ("service en chambre", "Please send room service"),
        ("ampoule", "The light bulb is broken"),
        ("depart tardif", "I need a late checkout"),
        ("commande de diner", "My dinner order is missing"),
    ],
    "de": [
        ("handtuch", "I need fresh towels"),
        ("klimaanlage", "The AC is not working"),
        ("zimmerservice", "Please send room service"),
        ("gluhbirne", "The light bulb is broken"),
        ("check out", "I need a late checkout"),
        ("essensbestellung", "My dinner order is missing"),
    ],
    "it": [
        ("asciugaman", "I need fresh towels"),
        ("aria condizionata", "The AC is not working"),
        ("servizio in camera", "Please send room service"),
        ("lampadina", "The light bulb is broken"),
        ("check out", "I need a late checkout"),
        ("ordine di cena", "My dinner order is missing"),
    ],
    "es": [
        ("toalla", "I need fresh towels"),
        ("aire acondicionado", "The AC is not working"),
        ("servicio a la habitacion", "Please send room service"),
        ("bombilla", "The light bulb is broken"),
        ("check out", "I need a late checkout"),
        ("pedido de cena", "My dinner order is missing"),
    ],
}

try:
    from faster_whisper import WhisperModel as FasterWhisperModel
    WhisperModel = FasterWhisperModel
except Exception as exc:  # pragma: no cover - environment-specific dependency issue
    _whisper_import_error = exc

try:
    from transformers import pipeline
    TranslationPipeline = pipeline
except Exception as exc:
    print(f"[WARN] transformers not available for translation: {exc}")

try:
    from langdetect import detect
except Exception:
    detect = None

try:
    from googletrans import Translator
    _google_translator = Translator()
except Exception as e:
    print(f"[WARN] googletrans not available: {e}")
    _google_translator = None

# Load faster-whisper model (tiny for speed, base for accuracy)
# tiny = ~39M, base = ~140M
_model = None


def normalize_repeated_text(text: str) -> str:
    """Collapse obvious contiguous duplicate runs in transcription or model output."""
    if not text:
        return ""

    normalized = " ".join(str(text).split()).strip()
    if not normalized:
        return ""

    tokens = normalized.split(" ")
    token_count = len(tokens)

    for block_size in range(1, token_count // 2 + 1):
        if token_count % block_size != 0:
            continue

        block = tokens[:block_size]
        repeats = token_count // block_size
        if block * repeats == tokens:
            return " ".join(block)

    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", normalized) if part.strip()]
    if len(sentences) > 1:
        collapsed_sentences = []
        for sentence in sentences:
            if not collapsed_sentences or collapsed_sentences[-1].lower() != sentence.lower():
                collapsed_sentences.append(sentence)
        normalized = " ".join(collapsed_sentences).strip()

    return normalized


def _get_env_config() -> dict:
    language = os.getenv("NEXORA_WHISPER_LANGUAGE", "auto").strip().lower()
    return {
        "model": os.getenv("NEXORA_WHISPER_MODEL", "base").strip() or "base",
        "device": os.getenv("NEXORA_WHISPER_DEVICE", "cpu").strip() or "cpu",
        "compute_type": os.getenv("NEXORA_WHISPER_COMPUTE_TYPE", "int8").strip() or "int8",
        "beam_size": int(os.getenv("NEXORA_WHISPER_BEAM_SIZE", "5")),
        "best_of": int(os.getenv("NEXORA_WHISPER_BEST_OF", "5")),
        "temperature": float(os.getenv("NEXORA_WHISPER_TEMPERATURE", "0.0")),
        "repetition_penalty": float(os.getenv("NEXORA_WHISPER_REPETITION_PENALTY", "1.1")),
        "no_repeat_ngram_size": int(os.getenv("NEXORA_WHISPER_NO_REPEAT_NGRAM_SIZE", "3")),
        "condition_on_previous_text": str(os.getenv("NEXORA_WHISPER_CONDITION_ON_PREVIOUS_TEXT", "false")).strip().lower() in {"1", "true", "yes", "on"},
        "task": os.getenv("NEXORA_WHISPER_TASK", "transcribe").strip() or "transcribe",
        "language": None if language in {"", "auto"} else language,
    }


def get_whisper_model():
    """Lazy-load whisper model."""
    global _model
    if _model is None:
        if WhisperModel is None:
            raise RuntimeError(
                "faster-whisper is not available in this environment. "
                "Install backend requirements in a Python 3.10 venv."
            ) from _whisper_import_error
        config = _get_env_config()
        print(
            "[INFO] Loading faster-whisper model "
            f"({config['model']}, device={config['device']}, compute_type={config['compute_type']})..."
        )
        _model = WhisperModel(
            config["model"],
            device=config["device"],
            compute_type=config["compute_type"],
        )
    return _model


def warmup_whisper_model() -> dict:
    """Load whisper model ahead of first transcription request."""
    config = _get_env_config()
    get_whisper_model()
    return {
        "status": "ready",
        "model": config["model"],
        "device": config["device"],
        "compute_type": config["compute_type"],
        "language": config["language"] or "auto",
        "best_of": config["best_of"],
        "temperature": config["temperature"],
        "repetition_penalty": config["repetition_penalty"],
        "no_repeat_ngram_size": config["no_repeat_ngram_size"],
        "condition_on_previous_text": config["condition_on_previous_text"],
    }


def _get_translator(src_lang: str, tgt_lang: str = "en"):
    """Get or create a translator pipeline for language pair."""
    if TranslationPipeline is None:
        return None
    
    cache_key = f"{src_lang}_to_{tgt_lang}"
    
    if cache_key in _translator_cache:
        return _translator_cache[cache_key]
    
    try:
        # Use Helsinki-NLP models which are lightweight and work offline
        model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
        print(f"[INFO] Loading translation model: {model_name}")
        translator = TranslationPipeline("translation", model=model_name)
        _translator_cache[cache_key] = translator
        return translator
    except Exception as e:
        print(f"[WARN] Could not load translator for {src_lang}->{tgt_lang}: {e}")
        _translator_cache[cache_key] = None
        return None


def _normalize_translation_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    normalized = "".join(character for character in normalized if not unicodedata.combining(character))
    normalized = normalized.lower().strip()
    normalized = re.sub(r"[^\w\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def _fallback_translate_phrase(text: str, src_lang: str, tgt_lang: str = "en") -> str:
    if tgt_lang.lower() != "en":
        return ""

    normalized = _normalize_translation_text(text)
    language = src_lang.lower()
    candidates = _PHRASE_TRANSLATIONS.get(language, {})
    if normalized in candidates:
        return candidates[normalized]

    for source_phrase, translated_phrase in candidates.items():
        if source_phrase in normalized:
            return translated_phrase

    for keyword, translated_phrase in _PHRASE_TRANSLATION_RULES.get(language, []):
        if keyword in normalized:
            return translated_phrase

    return ""


def _transcribe_with_options(model, audio_path: str, config: dict, *, language=None, vad_filter: bool = True) -> dict:
    """Run a single Whisper transcription pass and normalize the output."""
    segments, info = model.transcribe(
        audio_path,
        language=language if language is not None else config["language"],
        beam_size=config["beam_size"],
        best_of=config.get("best_of", 5),
        temperature=config.get("temperature", 0.0),
        repetition_penalty=config.get("repetition_penalty", 1.1),
        no_repeat_ngram_size=config.get("no_repeat_ngram_size", 3),
        condition_on_previous_text=config.get("condition_on_previous_text", False),
        task=config["task"],
        vad_filter=vad_filter,
    )
    segments_list = list(segments)

    text = normalize_repeated_text(" ".join([s.text for s in segments_list]).strip())

    confidences = []
    for segment in segments_list:
        no_speech_prob = getattr(segment, "no_speech_prob", None)
        if no_speech_prob is not None:
            confidences.append(max(0.0, min(1.0, 1.0 - float(no_speech_prob))))

    if confidences:
        confidence = sum(confidences) / len(confidences)
    else:
        confidence = max(0.0, min(1.0, float(getattr(info, "language_probability", 0.0) or 0.0)))

    detected_language = getattr(info, "language", None) or language or config["language"] or "unknown"
    return {
        "text": text,
        "confidence": round(confidence, 3),
        "language": detected_language,
        "segments": len(segments_list),
    }


def _write_temp_audio(audio_data: bytes) -> str:
    """Persist request audio to a temporary .webm file and return its path."""
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        tmp.write(audio_data)
        return tmp.name


def probe_audio_language(audio_data: bytes, language_hint: Optional[str] = None) -> Dict[str, Any]:
    """
    Lightweight audio language probe used before final STT profile selection.

    Returns:
        {
            "language": "en",
            "confidence": 0.82,
            "status": "success",
            "source": "whisper_probe"
        }
    """
    try:
        config = _get_env_config()
        model = get_whisper_model()
        tmp_path = _write_temp_audio(audio_data)
        try:
            _, info = model.transcribe(
                tmp_path,
                language=language_hint or None,
                beam_size=1,
                task=config["task"],
                vad_filter=False,
            )
            detected_language = getattr(info, "language", None) or "unknown"
            confidence = max(0.0, min(1.0, float(getattr(info, "language_probability", 0.0) or 0.0)))
            return {
                "language": detected_language,
                "confidence": round(confidence, 3),
                "status": "success",
                "source": "whisper_probe",
            }
        finally:
            os.unlink(tmp_path)
    except Exception as exc:
        return {
            "language": "unknown",
            "confidence": 0.0,
            "status": "error",
            "source": "whisper_probe",
            "error": str(exc),
        }


def translate_text(text: str, src_lang: str, tgt_lang: str = "en") -> str:
    """Translate text from source language to target language."""
    if not text or not text.strip():
        return ""
    
    # Auto-detect source language when requested
    if not src_lang or src_lang.lower() in {"auto", ""}:
        if detect is None:
            print("[WARN] langdetect not available to auto-detect language")
            return ""
        try:
            src_lang = detect(text)
            print(f"[INFO] Auto-detected language: {src_lang}")
        except Exception as e:
            print(f"[WARN] Language detection failed: {e}")
            return ""

    if src_lang.lower() == tgt_lang.lower():
        return text

    fallback_translation = _fallback_translate_phrase(text, src_lang, tgt_lang)
    if fallback_translation:
        print(f"[INFO] Phrase fallback translation: {src_lang} -> {tgt_lang}")
        return fallback_translation

    # Try language-specific Helsinki-NLP model first so the request is routed
    # through a dedicated translation model when one exists.
    translator = _get_translator(src_lang.lower(), tgt_lang.lower())
    if translator is not None:
        try:
            result = translator(text, max_length=512)
            if result and len(result) > 0:
                print(f"[INFO] Helsinki-NLP translation: {src_lang} -> {tgt_lang}")
                return result[0].get("translation_text", "").strip()
        except Exception as e:
            print(f"[WARN] Helsinki-NLP translation failed: {e}")

    # Fallback to googletrans if the model-based translator is unavailable.
    if _google_translator is not None:
        try:
            result = _google_translator.translate(text, dest=tgt_lang, src=src_lang)
            if result and hasattr(result, 'text') and result.text:
                print(f"[INFO] googletrans translation: {result.src} -> {result.dest}")
                return result.text.strip()
        except Exception as e:
            print(f"[WARN] googletrans failed: {e}")

    print(f"[WARN] Translation not available for {src_lang}->{tgt_lang}")
    return text.strip()


async def transcribe_audio(
    audio_data: bytes,
    profile: Optional[Dict[str, Any]] = None,
    probe_result: Optional[Dict[str, Any]] = None,
) -> dict:
    """
    Transcribe audio bytes to text using faster-whisper.
    Auto-translates to English if a non-English language is detected.
    
    Args:
        audio_data: Raw audio bytes (WAV, MP3, etc.)
    
    Returns:
        {
            "text": "...",
            "confidence": 0.95,
            "language": "es",
            "status": "success",
            "translation": "... english version ..." (if language != English)
            "translation_language": "en" (if translation provided)
        }
    """
    try:
        config = _get_env_config()
        profile = dict(profile or {})
        profile_model = str(profile.get("model") or config["model"]).strip() or config["model"]
        profile_language = profile.get("language")
        profile_beam_size = int(profile.get("beam_size", config["beam_size"]))
        profile_vad_filter = bool(profile.get("vad_filter", True))

        if profile_model != config["model"]:
            print(
                f"[WARN] Requested profile model '{profile_model}' differs from loaded env model "
                f"'{config['model']}'. Using loaded model."
            )

        effective_config = dict(config)
        effective_config["beam_size"] = profile_beam_size
        model = get_whisper_model()
        
        # Write audio to a temp file with a matching browser recording suffix.
        tmp_path = _write_temp_audio(audio_data)
        
        print(f"[INFO] Transcribing audio file: {tmp_path} ({len(audio_data)} bytes)")
        
        try:
            hinted_language = None
            if profile_language:
                hinted_language = str(profile_language).strip().lower()
            elif probe_result and probe_result.get("status") == "success":
                candidate_lang = str(probe_result.get("language", "")).strip().lower()
                candidate_conf = float(probe_result.get("confidence", 0.0) or 0.0)
                if candidate_lang and candidate_lang not in {"unknown", "und", "none"} and candidate_conf >= 0.55:
                    hinted_language = candidate_lang

            primary = _transcribe_with_options(
                model,
                tmp_path,
                effective_config,
                language=hinted_language if hinted_language else effective_config["language"],
                vad_filter=profile_vad_filter,
            )
            print(
                f"[INFO] Primary transcription: {primary['segments']} segments, "
                f"{len(primary['text'])} chars, confidence: {primary['confidence']}"
            )

            result = {
                "text": primary["text"],
                "confidence": primary["confidence"],
                "language": primary["language"],
                "status": "success",
                "segments": primary["segments"],
                "transcription_profile": {
                    "model": profile_model,
                    "language": profile_language,
                    "beam_size": profile_beam_size,
                    "vad_filter": profile_vad_filter,
                    "probe_language_hint": hinted_language,
                },
            }

            if len(primary["text"]) < 3:
                retry_candidates = [
                    (None, False, "auto_no_vad"),
                    ("th", False, "thai_no_vad"),
                ]
                best_retry = primary

                for language_hint, vad_filter, label in retry_candidates:
                    retry = _transcribe_with_options(
                        model,
                        tmp_path,
                        effective_config,
                        language=language_hint,
                        vad_filter=vad_filter,
                    )
                    print(
                        f"[INFO] Retry {label}: {retry['segments']} segments, "
                        f"{len(retry['text'])} chars, confidence: {retry['confidence']}"
                    )

                    if len(retry["text"]) > len(best_retry["text"]) or retry["confidence"] > best_retry["confidence"]:
                        best_retry = retry

                    if len(best_retry["text"]) >= 3:
                        break

                result["text"] = best_retry["text"]
                result["confidence"] = best_retry["confidence"]
                result["language"] = best_retry["language"]
                result["segments"] = best_retry["segments"]

            result["text"] = normalize_repeated_text(result["text"])
            print(f"[INFO] Transcription complete: {len(result['text'])} chars, confidence: {result['confidence']}")
            
            # Translate to English if language is not English.
            # Translation is best-effort only; do not fail transcription if it breaks.
            detected_language = result["language"]
            if detected_language and detected_language.lower() not in {"en", "english"}:
                try:
                    print(f"[INFO] Detected language: {detected_language}, translating to English...")
                    translation = translate_text(result["text"].strip(), detected_language, "en")
                    if translation:
                        cleaned_translation = normalize_repeated_text(translation)
                        print(f"[INFO] Translation: {cleaned_translation}")
                        result["translation"] = cleaned_translation
                        result["translation_language"] = "en"
                except Exception as translation_error:
                    print(f"[WARN] Translation skipped after transcription: {translation_error}")
            
            return result
        finally:
            # Clean up temp file
            os.unlink(tmp_path)
    
    except Exception as e:
        print(f"[ERROR] Transcription failed: {str(e)}")
        return {
            "text": "",
            "error": str(e),
            "status": "error"
        }
