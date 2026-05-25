import json
import os
import re
import unicodedata
from typing import Dict, Any

from app.schemas import IntentExtractionResult


INTENT_KEYWORDS = {
    "towel_request": [
        "towel",
        "towels",
        "extra towel",
        "serviette",
        "serviettes",
        "handtuch",
        "handtucher",
        "handtuecher",
        "asciugamano",
        "asciugamani",
        "toalla",
        "toallas",
    ],
    "food_order": [
        "food",
        "breakfast",
        "lunch",
        "dinner",
        "pizza",
        "burger",
        "fries",
        "drink",
        "drinks",
        "commande de dîner",
        "bestellung",
        "ordine di cena",
        "pedido de cena",
    ],
    "maintenance_request": [
        "fix",
        "broken",
        "repair",
        "leak",
        "light",
        "ac",
        "air conditioner",
        "toilet",
        "sink",
        "climatisation",
        "climatisation ne fonctionne pas",
        "klimaanlage",
        "klimaanlage funktioniert nicht",
        "aria condizionata",
        "aria condizionata non funziona",
        "aire acondicionado",
        "aire acondicionado no funciona",
        "ampoule",
        "lampadina",
        "bombilla",
    ],
    "room_service": [
        "room service",
        "send",
        "bring",
        "deliver",
        "service en chambre",
        "zimmerservice",
        "servizio in camera",
        "servicio a la habitacion",
    ],
    "housekeeping": [
        "clean",
        "cleaning",
        "housekeeping",
        "make up the room",
        "make up room",
        "menage",
        "zimmer reinigen",
        "pulizia",
        "limpio",
        "pulito",
    ],
}

INTENT_TO_DEPARTMENT = {
    "towel_request": "housekeeping",
    "food_order": "kitchen",
    "maintenance_request": "maintenance",
    "room_service": "room_service",
    "housekeeping": "housekeeping",
    "general_request": "front_desk",
}


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    normalized = "".join(character for character in normalized if not unicodedata.combining(character))
    normalized = normalized.lower().strip()
    normalized = re.sub(r"[^\w\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def _detect_quantity(text: str) -> int | None:
    match = re.search(r"\b(\d+)\b", text)
    if match:
        return int(match.group(1))
    if any(word in text for word in ["one", "single"]):
        return 1
    if any(word in text for word in ["two", "double"]):
        return 2
    if any(word in text for word in ["three", "triple"]):
        return 3
    return None


def _detect_items(text: str, intent: str) -> list[str]:
    if intent == "towel_request":
        return ["towel"]
    if intent == "food_order":
        items = []
        for keyword in ["breakfast", "lunch", "dinner", "pizza", "burger", "fries", "drink", "coffee", "tea"]:
            if keyword in text:
                items.append(keyword)
        return items or ["food"]
    if intent == "maintenance_request":
        items = []
        for keyword in ["light", "ac", "air conditioner", "toilet", "sink", "door", "heater", "leak"]:
            if keyword in text:
                items.append(keyword)
        return items or ["maintenance issue"]
    if intent == "housekeeping":
        return ["housekeeping help"]
    if intent == "room_service":
        return ["room service request"]
    return []


def _heuristic_extract(text: str) -> Dict[str, Any]:
    normalized = _normalize_text(text)
    selected_intent = "general_request"
    confidence = 0.55

    for intent, keywords in INTENT_KEYWORDS.items():
        if any(keyword in normalized for keyword in keywords):
            selected_intent = intent
            confidence = 0.91 if intent in ("towel_request", "maintenance_request") else 0.84
            break

    department = INTENT_TO_DEPARTMENT[selected_intent]
    quantity = _detect_quantity(normalized)
    items = _detect_items(normalized, selected_intent)

    return {
        "intent": selected_intent,
        "department": department,
        "items": items,
        "quantity": quantity,
        "confidence": confidence if selected_intent != "general_request" else 0.45,
        "needs_confirmation": True,
        "raw_text": text,
        "source": "heuristic",
    }


def _try_ollama_extract(text: str) -> Dict[str, Any] | None:
    if os.getenv("NEXORA_LLM_PROVIDER", "").lower() != "ollama":
        return None

    model = os.getenv("NEXORA_OLLAMA_MODEL", "llama3.2:3b")
    prompt = f"""
Return ONLY valid JSON with these keys: intent, department, items, quantity, confidence, needs_confirmation.

Allowed intents: towel_request, food_order, maintenance_request, room_service, housekeeping, general_request
Allowed departments: housekeeping, kitchen, maintenance, front_desk, room_service

Rules:
- confidence must be 0.0 to 1.0
- use items as a JSON array of strings
- quantity should be null or a number
- needs_confirmation must be true
- do not add any extra keys

Text: {text}
""".strip()

    try:
        import urllib.request

        payload = json.dumps(
            {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=20) as response:
            data = json.loads(response.read().decode("utf-8"))
        raw = data.get("response", "{}")
        return json.loads(raw)
    except Exception:
        return None


def extract_intent(text: str) -> IntentExtractionResult:
    candidate = _try_ollama_extract(text) or _heuristic_extract(text)
    result = IntentExtractionResult.parse_obj(candidate)

    if result.confidence < 0.6:
        return result.copy(update={"source": f"{result.source}-low-confidence"})

    return result
