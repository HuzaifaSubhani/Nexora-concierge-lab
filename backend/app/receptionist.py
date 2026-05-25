"""Prompt-aligned receptionist rules for the voice-only call flow."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

try:
    from langdetect import detect_langs
except Exception:  # pragma: no cover - optional dependency in minimal envs
    detect_langs = None


SUPPORTED_LANGUAGE_CODES = ("en", "fr", "it", "de", "nl", "es")

LANGUAGE_NAMES: Dict[str, str] = {
    "en": "English",
    "fr": "French",
    "it": "Italian",
    "de": "German",
    "nl": "Dutch",
    "es": "Spanish",
}

TTS_LANGUAGES: Dict[str, str] = {
    "en": "en",
    "fr": "fr",
    "it": "it",
    "de": "de",
    "nl": "nl",
    "es": "es",
}

INITIAL_GREETING = "Hello, welcome. Please speak in your preferred language."

LANGUAGE_CONFIRMATION_PROMPTS: Dict[str, str] = {
    "en": "I detected English. Would you like to continue in English?",
    "fr": "J'ai d\u00e9tect\u00e9 le fran\u00e7ais. Voulez-vous continuer en fran\u00e7ais?",
    "it": "Ho rilevato l'italiano. Vuole continuare in italiano?",
    "de": "Ich habe Deutsch erkannt. M\u00f6chten Sie auf Deutsch fortfahren?",
    "nl": "Ik heb Nederlands gedetecteerd. Wilt u doorgaan in het Nederlands?",
    "es": "He detectado espa\u00f1ol. \u00bfDesea continuar en espa\u00f1ol?",
}

ASK_LANGUAGE_PROMPT = "Okay. Which language would you like to continue in?"
UNSUPPORTED_LANGUAGE_PROMPT = (
    "Sorry, I currently support English, French, Italian, German, Dutch, and Spanish. "
    "Which one would you prefer?"
)
LANGUAGE_DETECTION_FAILED_PROMPT = (
    "Sorry, I could not detect the language. Please speak again in English, French, "
    "Italian, German, Dutch, or Spanish."
)

QUERY_PROMPTS: Dict[str, str] = {
    "en": "How can I help you today?",
    "fr": "Comment puis-je vous aider aujourd'hui?",
    "it": "Come posso aiutarla oggi?",
    "de": "Wie kann ich Ihnen heute helfen?",
    "nl": "Waarmee kan ik u vandaag helpen?",
    "es": "\u00bfC\u00f3mo puedo ayudarle hoy?",
}

GOODBYE_MESSAGES: Dict[str, str] = {
    "en": "Thank you for calling. Have a great day.",
    "fr": "Merci pour votre appel. Bonne journ\u00e9e.",
    "it": "Grazie per la chiamata. Buona giornata.",
    "de": "Vielen Dank f\u00fcr Ihren Anruf. Einen sch\u00f6nen Tag noch.",
    "nl": "Bedankt voor uw telefoontje. Fijne dag.",
    "es": "Gracias por llamar. Que tenga un buen d\u00eda.",
}

NO_SPEECH_MESSAGES: Dict[str, str] = {
    "en": "Sorry, I didn't hear anything. Are you still there?",
    "fr": "D\u00e9sol\u00e9, je n'ai rien entendu. \u00cates-vous toujours l\u00e0?",
    "it": "Mi dispiace, non ho sentito nulla. \u00c8 ancora in linea?",
    "de": "Entschuldigung, ich habe nichts geh\u00f6rt. Sind Sie noch da?",
    "nl": "Sorry, ik heb niets gehoord. Bent u er nog?",
    "es": "Lo siento, no he o\u00eddo nada. \u00bfSigue ah\u00ed?",
}

ROUTING_CLARIFICATION_MESSAGES: Dict[str, str] = {
    "en": "Sorry, I could not understand clearly. Are you calling for appointments, billing, support, or general information?",
    "fr": "D\u00e9sol\u00e9, je n'ai pas bien compris. Appelez-vous pour les rendez-vous, la facturation, le support ou des informations g\u00e9n\u00e9rales?",
    "it": "Mi dispiace, non ho capito bene. Chiama per appuntamenti, fatturazione, supporto o informazioni generali?",
    "de": "Entschuldigung, ich habe das nicht ganz verstanden. Rufen Sie wegen Terminen, Abrechnung, Support oder allgemeinen Informationen an?",
    "nl": "Sorry, ik heb het niet goed begrepen. Belt u voor afspraken, facturatie, support of algemene informatie?",
    "es": "Lo siento, no he entendido bien. \u00bfLlama por citas, facturaci\u00f3n, soporte o informaci\u00f3n general?",
}

HUMAN_AFTER_FAILURE_MESSAGES: Dict[str, str] = {
    "en": "I'm sorry, I'm still having trouble understanding. I will note that you need a human representative.",
    "fr": "Je suis d\u00e9sol\u00e9, j'ai encore du mal \u00e0 comprendre. Je vais noter que vous avez besoin d'un repr\u00e9sentant humain.",
    "it": "Mi dispiace, faccio ancora fatica a capire. Annoter\u00f2 che ha bisogno di un rappresentante umano.",
    "de": "Es tut mir leid, ich habe immer noch Schwierigkeiten, Sie zu verstehen. Ich notiere, dass Sie einen menschlichen Mitarbeiter brauchen.",
    "nl": "Het spijt me, ik begrijp het nog steeds niet goed. Ik noteer dat u een menselijke medewerker nodig heeft.",
    "es": "Lo siento, todav\u00eda tengo problemas para entenderle. Anotar\u00e9 que necesita un representante humano.",
}

OUT_OF_SCOPE_MESSAGES: Dict[str, str] = {
    "en": "I'm sorry, I can only help with appointments, billing, support, general information, or a human representative.",
    "fr": "D\u00e9sol\u00e9, je peux seulement aider avec les rendez-vous, la facturation, le support, les informations g\u00e9n\u00e9rales ou un repr\u00e9sentant humain.",
    "it": "Mi dispiace, posso aiutare solo con appuntamenti, fatturazione, supporto, informazioni generali o un rappresentante umano.",
    "de": "Entschuldigung, ich kann nur bei Terminen, Abrechnung, Support, allgemeinen Informationen oder einem menschlichen Mitarbeiter helfen.",
    "nl": "Sorry, ik kan alleen helpen met afspraken, facturatie, support, algemene informatie of een menselijke medewerker.",
    "es": "Lo siento, solo puedo ayudar con citas, facturaci\u00f3n, soporte, informaci\u00f3n general o un representante humano.",
}

PROFESSIONAL_ADVICE_MESSAGES: Dict[str, str] = {
    "en": "I'm sorry, I cannot provide professional advice. I can note that you need the right department or a human representative.",
    "fr": "D\u00e9sol\u00e9, je ne peux pas fournir de conseil professionnel. Je peux noter que vous avez besoin du bon service ou d'un repr\u00e9sentant humain.",
    "it": "Mi dispiace, non posso fornire consulenza professionale. Posso annotare che ha bisogno del reparto giusto o di un rappresentante umano.",
    "de": "Entschuldigung, ich kann keine fachliche Beratung geben. Ich kann notieren, dass Sie die richtige Abteilung oder einen menschlichen Mitarbeiter brauchen.",
    "nl": "Sorry, ik kan geen professioneel advies geven. Ik kan noteren dat u de juiste afdeling of een menselijke medewerker nodig heeft.",
    "es": "Lo siento, no puedo dar asesoramiento profesional. Puedo anotar que necesita el departamento adecuado o un representante humano.",
}

ANGER_MESSAGES: Dict[str, str] = {
    "en": "I'm sorry for the inconvenience. I will note that you need a human representative.",
    "fr": "Je suis d\u00e9sol\u00e9 pour ce d\u00e9sagr\u00e9ment. Je vais noter que vous avez besoin d'un repr\u00e9sentant humain.",
    "it": "Mi dispiace per l'inconveniente. Annoter\u00f2 che ha bisogno di un rappresentante umano.",
    "de": "Es tut mir leid wegen der Unannehmlichkeit. Ich notiere, dass Sie einen menschlichen Mitarbeiter brauchen.",
    "nl": "Het spijt me voor het ongemak. Ik noteer dat u een menselijke medewerker nodig heeft.",
    "es": "Siento las molestias. Anotar\u00e9 que necesita un representante humano.",
}

DEPARTMENT_RESPONSES_EN: Dict[str, str] = {
    "Appointments": "Sure, I will note that you need the appointments department. Please keep your preferred date and time ready.",
    "Billing": "I understand. I will note that you need the billing department. Please keep your invoice or payment details ready.",
    "Support": "I understand you need help. I will note that you need the support department.",
    "General Information": "Sure, I can help with general information. Please tell me what you would like to know.",
    "Human Transfer": "Sure, I will note that you need to speak with a human representative.",
}

DEPARTMENT_RESPONSES: Dict[str, Dict[str, str]] = {
    "en": DEPARTMENT_RESPONSES_EN,
    "fr": {
        "Appointments": "Bien s\u00fbr, je vais noter que vous avez besoin du service des rendez-vous. Veuillez pr\u00e9parer la date et l'heure qui vous conviennent.",
        "Billing": "Je comprends. Je vais noter que vous avez besoin du service de facturation. Veuillez pr\u00e9parer votre facture ou vos d\u00e9tails de paiement.",
        "Support": "Je comprends que vous avez besoin d'aide. Je vais noter que vous avez besoin du service support.",
        "General Information": "Bien s\u00fbr, je peux aider avec les informations g\u00e9n\u00e9rales. Dites-moi ce que vous souhaitez savoir.",
        "Human Transfer": "Bien s\u00fbr, je vais noter que vous souhaitez parler \u00e0 un repr\u00e9sentant humain.",
    },
    "it": {
        "Appointments": "Certo, annoter\u00f2 che ha bisogno del reparto appuntamenti. Tenga pronta la data e l'orario preferiti.",
        "Billing": "Capisco. Annoter\u00f2 che ha bisogno del reparto fatturazione. Tenga pronti la fattura o i dettagli del pagamento.",
        "Support": "Capisco che ha bisogno di aiuto. Annoter\u00f2 che ha bisogno del reparto supporto.",
        "General Information": "Certo, posso aiutare con le informazioni generali. Mi dica cosa desidera sapere.",
        "Human Transfer": "Certo, annoter\u00f2 che desidera parlare con un rappresentante umano.",
    },
    "de": {
        "Appointments": "Gerne, ich notiere, dass Sie die Terminabteilung ben\u00f6tigen. Bitte halten Sie Ihr bevorzugtes Datum und Ihre Uhrzeit bereit.",
        "Billing": "Ich verstehe. Ich notiere, dass Sie die Abrechnungsabteilung ben\u00f6tigen. Bitte halten Sie Ihre Rechnung oder Zahlungsdaten bereit.",
        "Support": "Ich verstehe, dass Sie Hilfe brauchen. Ich notiere, dass Sie den Support ben\u00f6tigen.",
        "General Information": "Gerne, ich kann bei allgemeinen Informationen helfen. Bitte sagen Sie mir, was Sie wissen m\u00f6chten.",
        "Human Transfer": "Gerne, ich notiere, dass Sie mit einem menschlichen Mitarbeiter sprechen m\u00f6chten.",
    },
    "nl": {
        "Appointments": "Natuurlijk, ik noteer dat u de afsprakenafdeling nodig heeft. Houd uw gewenste datum en tijd bij de hand.",
        "Billing": "Ik begrijp het. Ik noteer dat u de facturatieafdeling nodig heeft. Houd uw factuur of betaalgegevens bij de hand.",
        "Support": "Ik begrijp dat u hulp nodig heeft. Ik noteer dat u support nodig heeft.",
        "General Information": "Natuurlijk, ik kan helpen met algemene informatie. Vertel me wat u wilt weten.",
        "Human Transfer": "Natuurlijk, ik noteer dat u met een menselijke medewerker wilt spreken.",
    },
    "es": {
        "Appointments": "Claro, anotar\u00e9 que necesita el departamento de citas. Tenga preparada la fecha y hora que prefiera.",
        "Billing": "Entiendo. Anotar\u00e9 que necesita el departamento de facturaci\u00f3n. Tenga preparada su factura o los detalles de pago.",
        "Support": "Entiendo que necesita ayuda. Anotar\u00e9 que necesita el departamento de soporte.",
        "General Information": "Claro, puedo ayudarle con informaci\u00f3n general. D\u00edgame qu\u00e9 desea saber.",
        "Human Transfer": "Claro, anotar\u00e9 que desea hablar con un representante humano.",
    },
}

LANGUAGE_SIGNAL_KEYWORDS: Dict[str, Iterable[str]] = {
    "en": ("hello", "hi", "good morning", "good afternoon", "english"),
    "fr": ("bonjour", "bonsoir", "salut", "francais", "fran\u00e7ais"),
    "it": ("ciao", "buongiorno", "buonasera", "italiano"),
    "de": ("hallo", "guten tag", "guten morgen", "deutsch"),
    "nl": ("hallo", "goedemorgen", "goedenavond", "nederlands"),
    "es": ("hola", "buenos dias", "buenos d\u00edas", "buenas tardes", "espanol", "espa\u00f1ol"),
}

LANGUAGE_SELECTION_KEYWORDS: Dict[str, Iterable[str]] = {
    "en": ("english", "anglais", "inglese", "englisch", "engels", "ingles"),
    "fr": ("french", "francais", "fran\u00e7ais", "francese", "franzoesisch", "franzosisch", "frans", "frances"),
    "it": ("italian", "italien", "italiano", "italienisch", "italiaans"),
    "de": ("german", "allemand", "tedesco", "deutsch", "duits", "aleman"),
    "nl": ("dutch", "neerlandais", "olandese", "niederlaendisch", "niederlandisch", "nederlands", "holandes"),
    "es": ("spanish", "espagnol", "spagnolo", "spanisch", "spaans", "espanol", "espa\u00f1ol"),
}

YES_WORDS: Dict[str, Iterable[str]] = {
    "en": ("yes", "yeah", "correct", "okay", "ok", "sure"),
    "fr": ("oui", "d accord", "daccord", "correct"),
    "it": ("si", "s\u00ec", "va bene", "corretto"),
    "de": ("ja", "genau", "richtig", "okay", "ok"),
    "nl": ("ja", "goed", "klopt", "oke", "ok\u00e9", "ok"),
    "es": ("si", "s\u00ed", "correcto", "esta bien", "est\u00e1 bien", "vale"),
}

NO_WORDS: Dict[str, Iterable[str]] = {
    "en": ("no", "nope", "wrong", "change", "different"),
    "fr": ("non", "changer", "incorrect"),
    "it": ("no", "cambia", "sbagliato"),
    "de": ("nein", "aendern", "\u00e4ndern", "falsch"),
    "nl": ("nee", "wijzigen", "verkeerd"),
    "es": ("no", "cambiar", "incorrecto"),
}

DEPARTMENT_KEYWORDS: Dict[str, Iterable[str]] = {
    "Human Transfer": (
        "human",
        "agent",
        "representative",
        "manager",
        "real person",
        "operator",
        "staff member",
        "speak to someone",
        "transfer me",
        "customer service",
    ),
    "Appointments": (
        "appointment",
        "booking",
        "schedule",
        "meeting",
        "reservation",
        "availability",
        "date",
        "time slot",
        "reschedule",
        "cancel appointment",
    ),
    "Billing": (
        "bill",
        "billing",
        "invoice",
        "payment",
        "refund",
        "charge",
        "receipt",
        "price",
        "cost",
        "transaction",
        "overdue payment",
    ),
    "Support": (
        "problem",
        "issue",
        "error",
        "not working",
        "technical help",
        "complaint",
        "broken",
        "failed",
        "cannot access",
        "need help",
    ),
    "General Information": (
        "address",
        "location",
        "opening hours",
        "closing time",
        "working days",
        "services",
        "company information",
        "office",
        "directions",
        "contact information",
    ),
}

MULTILINGUAL_INTENT_HINTS: Dict[str, Iterable[str]] = {
    "Human Transfer": (
        "humain",
        "representant",
        "repr\u00e9sentant",
        "personne",
        "operateur",
        "op\u00e9rateur",
        "parler a quelqu un",
        "parler \u00e0 quelqu'un",
        "operatore",
        "persona reale",
        "parlare con qualcuno",
        "mitarbeiter",
        "mensch",
        "kundendienst",
        "medewerker",
        "vertegenwoordiger",
        "iemand spreken",
        "representante",
        "persona real",
        "hablar con alguien",
        "servicio al cliente",
        "atencion al cliente",
        "atenci\u00f3n al cliente",
    ),
    "Appointments": (
        "rendez vous",
        "rendez-vous",
        "prendre rendez vous",
        "reservation",
        "r\u00e9servation",
        "disponibilite",
        "disponibilit\u00e9",
        "appuntamento",
        "prenotazione",
        "disponibilita",
        "disponibilit\u00e0",
        "termin",
        "reservierung",
        "verfugbarkeit",
        "verf\u00fcgbarkeit",
        "afspraak",
        "reservering",
        "beschikbaarheid",
        "cita",
        "reserva",
        "disponibilidad",
    ),
    "Billing": (
        "facture",
        "facturation",
        "paiement",
        "remboursement",
        "prix",
        "cout",
        "co\u00fbt",
        "fattura",
        "pagamento",
        "rimborso",
        "prezzo",
        "costo",
        "rechnung",
        "zahlung",
        "rueckerstattung",
        "ruckerstattung",
        "r\u00fcckerstattung",
        "kosten",
        "factuur",
        "betaling",
        "terugbetaling",
        "prijs",
        "factura",
        "pago",
        "reembolso",
        "recibo",
        "costo",
    ),
    "Support": (
        "probleme",
        "probl\u00e8me",
        "erreur",
        "ne fonctionne pas",
        "aide technique",
        "plainte",
        "cass\u00e9",
        "casse",
        "problema",
        "errore",
        "non funziona",
        "aiuto tecnico",
        "reclamo",
        "rotto",
        "fehler",
        "funktioniert nicht",
        "technische hilfe",
        "beschwerde",
        "kaputt",
        "probleem",
        "fout",
        "werkt niet",
        "technische hulp",
        "klacht",
        "defect",
        "error",
        "no funciona",
        "ayuda tecnica",
        "ayuda t\u00e9cnica",
        "queja",
        "roto",
    ),
    "General Information": (
        "adresse",
        "horaires",
        "heures d ouverture",
        "heures d'ouverture",
        "services",
        "itineraire",
        "itin\u00e9raire",
        "indirizzo",
        "orari",
        "servizi",
        "indicazioni",
        "offnungszeiten",
        "\u00f6ffnungszeiten",
        "dienstleistungen",
        "weg",
        "adres",
        "openingstijden",
        "diensten",
        "route",
        "direccion",
        "direcci\u00f3n",
        "horario",
        "servicios",
        "ubicacion",
        "ubicaci\u00f3n",
    ),
}

ADVICE_KEYWORDS = (
    "medical advice",
    "legal advice",
    "financial advice",
    "doctor",
    "lawyer",
    "investment",
    "conseil medical",
    "conseil m\u00e9dical",
    "conseil juridique",
    "consulenza medica",
    "consulenza legale",
    "medizinische beratung",
    "rechtsberatung",
    "medisch advies",
    "juridisch advies",
    "consejo medico",
    "consejo m\u00e9dico",
    "asesoramiento legal",
)

OUT_OF_SCOPE_KEYWORDS = (
    "weather",
    "news",
    "joke",
    "recipe",
    "homework",
    "story",
    "meteo",
    "m\u00e9t\u00e9o",
    "blague",
    "ricetta",
    "barzelletta",
    "wetter",
    "witz",
    "weer",
    "grap",
    "clima",
    "chiste",
)

ANGER_KEYWORDS = (
    "angry",
    "furious",
    "mad",
    "upset",
    "complain",
    "complaint",
    "colere",
    "col\u00e8re",
    "furieux",
    "furieuse",
    "plainte",
    "arrabbiato",
    "arrabbiata",
    "reclamo",
    "wutend",
    "w\u00fctend",
    "beschwerde",
    "boos",
    "klacht",
    "enojado",
    "enojada",
    "queja",
)

END_CALL_KEYWORDS = (
    "bye",
    "goodbye",
    "thank you bye",
    "au revoir",
    "arrivederci",
    "tschuss",
    "auf wiedersehen",
    "tot ziens",
    "adios",
    "adi\u00f3s",
)


@dataclass(frozen=True)
class LanguageDetection:
    code: str
    name: str
    confidence: float


@dataclass(frozen=True)
class RouteDecision:
    intent: str
    confidence: str
    response_english: str
    response_final_language: str
    action: str
    notes: str


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    normalized = "".join(character for character in normalized if not unicodedata.combining(character))
    normalized = normalized.lower().strip()
    normalized = re.sub(r"[^\w\s'-]", " ", normalized)
    normalized = normalized.replace("'", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _contains_any(text: str, keywords: Iterable[str]) -> bool:
    normalized = normalize_text(text)
    return any(normalize_text(keyword) in normalized for keyword in keywords)


def _language_from_keywords(text: str, mapping: Dict[str, Iterable[str]]) -> Optional[str]:
    normalized = normalize_text(text)
    for code, keywords in mapping.items():
        if normalized == code or any(normalize_text(keyword) in normalized for keyword in keywords):
            return code
    return None


def _language_from_selection(text: str) -> Optional[str]:
    return _language_from_keywords(text, LANGUAGE_SELECTION_KEYWORDS)


def detect_supported_language(text: str) -> LanguageDetection:
    selected = _language_from_selection(text)
    if selected:
        return LanguageDetection(selected, LANGUAGE_NAMES[selected], 100.0)

    signal = _language_from_keywords(text, LANGUAGE_SIGNAL_KEYWORDS)
    if signal:
        return LanguageDetection(signal, LANGUAGE_NAMES[signal], 92.0)

    if not text or len(text.strip()) < 3 or detect_langs is None:
        return LanguageDetection("unknown", "Unknown", 0.0)

    try:
        candidates = detect_langs(text)
    except Exception:
        return LanguageDetection("unknown", "Unknown", 0.0)

    for candidate in candidates:
        code = str(candidate.lang).split("-")[0].lower()
        if code in SUPPORTED_LANGUAGE_CODES:
            return LanguageDetection(code, LANGUAGE_NAMES[code], float(candidate.prob) * 100.0)

    return LanguageDetection("unknown", "Unknown", 0.0)


def confirmation_prompt_for_language(code: str) -> str:
    return LANGUAGE_CONFIRMATION_PROMPTS.get(code, LANGUAGE_DETECTION_FAILED_PROMPT)


def query_prompt_for_language(code: str) -> str:
    return QUERY_PROMPTS.get(code, QUERY_PROMPTS["en"])


def language_code_from_name_or_code(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    normalized = normalize_text(value)
    if normalized in SUPPORTED_LANGUAGE_CODES:
        return normalized
    return _language_from_selection(value)


def parse_confirmation(text: str, language_code: str) -> Optional[bool]:
    candidates_yes = list(YES_WORDS.get(language_code, ())) + list(YES_WORDS["en"])
    candidates_no = list(NO_WORDS.get(language_code, ())) + list(NO_WORDS["en"])
    if _contains_any(text, candidates_yes):
        return True
    if _contains_any(text, candidates_no):
        return False
    return None


def classify_intent(english_text: str, original_text: str = "") -> tuple[str, str, str]:
    text = f"{english_text} {original_text}".strip()
    if not text:
        return "Unknown", "low", "No speech detected."

    if _contains_any(text, ADVICE_KEYWORDS):
        return "Human Transfer", "medium", "Caller requested professional advice."

    if _contains_any(text, OUT_OF_SCOPE_KEYWORDS):
        return "Unknown", "low", "Caller asked outside the receptionist scope."

    for intent in ("Human Transfer", "Appointments", "Billing", "Support", "General Information"):
        if _contains_any(text, DEPARTMENT_KEYWORDS[intent]) or _contains_any(text, MULTILINGUAL_INTENT_HINTS[intent]):
            return intent, "high", f"Caller matches {intent.lower()} routing keywords."

    return "Unknown", "low", "No department keywords matched."


def english_meaning_from_keywords(original_text: str, language_code: str, fallback: str = "") -> str:
    intent, confidence, _ = classify_intent(fallback or original_text, original_text)
    if confidence != "low" and intent != "Unknown":
        return f"The caller needs {intent.lower()}."
    return fallback or original_text


def final_response_for_intent(intent: str, language_code: str) -> str:
    responses = DEPARTMENT_RESPONSES.get(language_code, DEPARTMENT_RESPONSES["en"])
    return responses.get(intent, ROUTING_CLARIFICATION_MESSAGES.get(language_code, ROUTING_CLARIFICATION_MESSAGES["en"]))


def route_receptionist_query(
    caller_text_original: str,
    caller_text_english: str,
    language_code: str,
    failure_count: int = 0,
) -> RouteDecision:
    original = caller_text_original or ""
    english = caller_text_english or original
    lang = language_code if language_code in SUPPORTED_LANGUAGE_CODES else "en"

    if not original.strip():
        if failure_count >= 2:
            return RouteDecision(
                intent="Human Transfer",
                confidence="low",
                response_english=HUMAN_AFTER_FAILURE_MESSAGES["en"],
                response_final_language=HUMAN_AFTER_FAILURE_MESSAGES[lang],
                action="transfer_human",
                notes="Repeated no speech.",
            )
        return RouteDecision(
            intent="Unknown",
            confidence="low",
            response_english=NO_SPEECH_MESSAGES["en"],
            response_final_language=NO_SPEECH_MESSAGES[lang],
            action="ask_clarification",
            notes="No speech detected.",
        )

    normalized = normalize_text(f"{english} {original}")
    if any(normalize_text(keyword) in normalized for keyword in END_CALL_KEYWORDS):
        return RouteDecision(
            intent="Unknown",
            confidence="high",
            response_english=GOODBYE_MESSAGES["en"],
            response_final_language=GOODBYE_MESSAGES[lang],
            action="end_call",
            notes="Caller ended the call.",
        )

    if _contains_any(normalized, ANGER_KEYWORDS):
        return RouteDecision(
            intent="Human Transfer",
            confidence="medium",
            response_english=ANGER_MESSAGES["en"],
            response_final_language=ANGER_MESSAGES[lang],
            action="transfer_human",
            notes="Caller may be upset.",
        )

    if _contains_any(normalized, ADVICE_KEYWORDS):
        return RouteDecision(
            intent="Human Transfer",
            confidence="medium",
            response_english=PROFESSIONAL_ADVICE_MESSAGES["en"],
            response_final_language=PROFESSIONAL_ADVICE_MESSAGES[lang],
            action="transfer_human",
            notes="Professional advice request.",
        )

    if _contains_any(normalized, OUT_OF_SCOPE_KEYWORDS):
        return RouteDecision(
            intent="Unknown",
            confidence="low",
            response_english=OUT_OF_SCOPE_MESSAGES["en"],
            response_final_language=OUT_OF_SCOPE_MESSAGES[lang],
            action="ask_clarification",
            notes="Outside business scope.",
        )

    intent, confidence, notes = classify_intent(english, original)

    if intent == "Unknown":
        if failure_count >= 2:
            return RouteDecision(
                intent="Human Transfer",
                confidence="low",
                response_english=HUMAN_AFTER_FAILURE_MESSAGES["en"],
                response_final_language=HUMAN_AFTER_FAILURE_MESSAGES[lang],
                action="transfer_human",
                notes="Repeated unclear routing.",
            )
        return RouteDecision(
            intent="Unknown",
            confidence="low",
            response_english=ROUTING_CLARIFICATION_MESSAGES["en"],
            response_final_language=ROUTING_CLARIFICATION_MESSAGES[lang],
            action="ask_clarification",
            notes=notes,
        )

    action = "transfer_human" if intent == "Human Transfer" else "route_department"
    return RouteDecision(
        intent=intent,
        confidence=confidence,
        response_english=DEPARTMENT_RESPONSES_EN[intent],
        response_final_language=final_response_for_intent(intent, lang),
        action=action,
        notes=notes,
    )
