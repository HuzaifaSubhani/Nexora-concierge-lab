"""
Language Identification and Routing Layer.
Handles language detection, confidence scoring, and fallback DTMF prompts.
"""
import logging
from typing import Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)

# Expanded supported languages
SUPPORTED_LANGUAGES = {
    "en": "English",
    "ar": "Arabic",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "zh": "Chinese",
    "ja": "Japanese",
    "pt": "Portuguese",
    "ru": "Russian",
    "hi": "Hindi",
    "it": "Italian",
    "ko": "Korean",
    "nl": "Dutch",
    "pl": "Polish",
    "tr": "Turkish",
    "th": "Thai",
    "vi": "Vietnamese",
    "id": "Indonesian",
    "ms": "Malay",
    "fil": "Filipino",
}

class LanguageDetectionLibrary:
    """Wrapper for language detection with multiple fallbacks."""
    
    _detector = None
    _whisper_model = None

    @classmethod
    def _detect_script_language(cls, text: str) -> Tuple[Optional[str], float]:
        """Use Unicode script cues before falling back to statistical detection."""
        script_map = [
            ("ar", [(0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)]),
            ("th", [(0x0E00, 0x0E7F)]),
            ("zh", [(0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0xF900, 0xFAFF)]),
            ("ja", [(0x3040, 0x309F), (0x30A0, 0x30FF), (0x31F0, 0x31FF)]),
            ("ko", [(0xAC00, 0xD7AF), (0x1100, 0x11FF), (0x3130, 0x318F)]),
            ("ru", [(0x0400, 0x04FF)]),
            ("he", [(0x0590, 0x05FF)]),
            ("hi", [(0x0900, 0x097F)]),
            ("vi", [(0x0102, 0x0103), (0x0110, 0x0111), (0x1EA0, 0x1EF9)]),
        ]

        def _in_ranges(code_point: int, ranges: list[tuple[int, int]]) -> bool:
            return any(start <= code_point <= end for start, end in ranges)

        counts = {language_code: 0 for language_code, _ in script_map}
        total_letters = 0

        for character in text:
            if not character.isalpha():
                continue

            code_point = ord(character)
            total_letters += 1
            for language_code, ranges in script_map:
                if _in_ranges(code_point, ranges):
                    counts[language_code] += 1
                    break

        if total_letters == 0:
            return None, 0.0

        best_language = None
        best_count = 0
        for language_code, count in counts.items():
            if count > best_count:
                best_language = language_code
                best_count = count

        if not best_language or best_count == 0:
            return None, 0.0

        ratio = best_count / total_letters
        if ratio < 0.35:
            return None, 0.0

        if best_language == "he":
            return "ar", min(99.0, 80.0 + ratio * 20.0)

        return best_language, min(99.0, 80.0 + ratio * 20.0)
    
    @classmethod
    def _get_detector(cls):
        """Lazy-load langdetect."""
        if cls._detector is None:
            try:
                from langdetect import detect, detect_langs
                cls._detector = (detect, detect_langs)
                logger.info("langdetect loaded successfully")
            except ImportError:
                logger.warning("langdetect not installed. Falling back to Whisper language detection.")
                cls._detector = (None, None)
        return cls._detector
    
    @classmethod
    def _try_whisper_detection(cls, text: str) -> Optional[str]:
        """Try to detect language using Whisper model."""
        try:
            from app.services import get_whisper_model
            model = get_whisper_model()
            # Whisper can detect language from audio, but we're working with text
            # So we'll use it as a secondary fallback if available
            logger.debug("Whisper model available but text-based detection not supported via Whisper")
            return None
        except Exception as e:
            logger.debug(f"Whisper detection unavailable: {e}")
            return None
    
    @classmethod
    def detect_language_with_confidence(cls, text: str, min_length: int = 10) -> Tuple[Optional[str], float]:
        """
        Detect language from text and return confidence score (0-100).
        Uses multiple detection strategies for robustness.
        
        Args:
            text: Text to detect language from
            min_length: Minimum text length to attempt detection
        
        Returns:
            Tuple of (language_code, confidence_score)
        """
        if not text or len(text.strip()) < min_length:
            logger.warning(f"Text too short for detection: {len(text.strip())} chars")
            return None, 0.0

        script_lang, script_confidence = cls._detect_script_language(text)
        if script_lang:
            logger.info(f"Detected language via script heuristic: {script_lang} ({script_confidence:.1f}% confidence)")
            return script_lang, script_confidence
        
        detect, detect_langs = cls._get_detector()
        
        if detect is None:
            logger.warning("Language detection library unavailable. Trying Whisper fallback...")
            result = cls._try_whisper_detection(text)
            if result:
                return result, 60.0
            logger.warning("All detection methods failed. Returning English fallback.")
            return "en", 40.0
        
        try:
            # Get all language probabilities
            lang_probs = detect_langs(text)
            if lang_probs:
                top_lang = lang_probs[0]
                lang_code = top_lang.lang
                confidence = top_lang.prob * 100  # Convert to 0-100 scale
                
                logger.info(f"Detected language: {lang_code} ({confidence:.1f}% confidence)")
                
                # Normalize lang_code if needed (e.g., zh-cn -> zh)
                if "-" in lang_code:
                    lang_code = lang_code.split("-")[0]
                
                # Check if language is supported
                if lang_code not in SUPPORTED_LANGUAGES:
                    logger.info(f"Detected language {lang_code} not in supported list: {list(SUPPORTED_LANGUAGES.keys())}")
                    # Don't penalize confidence, but keep the original detection
                    # This allows for future expansion without code changes
                    return lang_code, confidence * 0.9
                
                return lang_code, confidence
        except Exception as e:
            logger.error(f"Language detection error: {e}. Returning fallback.")
            return "en", 50.0
        
        return None, 0.0


class LanguageRouter:
    """
    Routes incoming calls based on detected language.
    Handles confidence thresholds, fallback prompts, and session locking.
    """
    
    CONFIDENCE_THRESHOLD = 70.0  # Minimum confidence to lock language (lowered for better UX)
    FALLBACK_PROMPT_TEMPLATE = {
        "en": "I'm sorry, I couldn't understand your language. Press 1 for English, 2 for Arabic.",
        "ar": "عذرا، لم أتمكن من فهم لغتك. اضغط 1 للإنجليزية، 2 للعربية.",
        "es": "Lo siento, no entendí tu idioma. Presiona 1 para inglés, 2 para árabe.",
        "fr": "Désolé, je n'ai pas compris votre langue. Appuyez sur 1 pour l'anglais, 2 pour l'arabe.",
    }
    
    def __init__(self):
        self.detector = LanguageDetectionLibrary()
    
    def detect_language_from_transcription(self, transcription: str) -> Tuple[str, float, bool]:
        """
        Detect language from transcribed text.
        
        Returns:
            Tuple of (language_code, confidence, should_lock)
            - language_code: Detected language (e.g., 'en', 'ar', 'es')
            - confidence: Confidence score (0-100)
            - should_lock: Whether language should be locked for session
        """
        detected_lang, confidence = self.detector.detect_language_with_confidence(transcription)
        
        should_lock = confidence >= self.CONFIDENCE_THRESHOLD
        
        logger.info(f"Language routing: {detected_lang} @ {confidence:.1f}% → lock={should_lock}")
        
        return detected_lang or "en", confidence, should_lock
    
    def get_fallback_prompt(self, detected_lang: str = "en") -> Tuple[str, dict]:
        """
        Get fallback DTMF prompt for language selection menu.
        
        Returns:
            Tuple of (prompt_text, dtmf_options_dict)
        """
        # Use detected language as base, fallback to English
        base_lang = detected_lang if detected_lang in self.FALLBACK_PROMPT_TEMPLATE else "en"
        
        prompt_text = self.FALLBACK_PROMPT_TEMPLATE.get(base_lang, self.FALLBACK_PROMPT_TEMPLATE["en"])
        
        dtmf_options = {
            "1": "en",
            "2": "ar",
        }
        
        return prompt_text, dtmf_options
    
    def route_by_dtmf_selection(self, dtmf_key: str) -> Optional[str]:
        """
        Route based on DTMF keypad input during fallback prompt.
        
        Args:
            dtmf_key: Single digit from DTMF (e.g., '1', '2')
        
        Returns:
            Language code or None if invalid
        """
        dtmf_map = {
            "1": "en",
            "2": "ar",
        }
        
        lang = dtmf_map.get(dtmf_key)
        if lang:
            logger.info(f"DTMF selection: {dtmf_key} -> Language: {lang}")
        else:
            logger.warning(f"Invalid DTMF key: {dtmf_key}")
        
        return lang


# Global singleton instance
_router_instance = None

def get_language_router() -> LanguageRouter:
    """Get or create singleton language router."""
    global _router_instance
    if _router_instance is None:
        _router_instance = LanguageRouter()
    return _router_instance
