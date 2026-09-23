"""Language helpers with an optional langdetect dependency."""

LANGUAGE_NAMES = {
    "en": "English",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "bn": "Bengali",
    "unknown": "Unknown",
}


def _fallback_detect(text: str) -> str:
    lowered = f" {text.casefold()} "
    markers = {
        "de": (" ich ", " mein ", " meine ", " passwort", "bestellung", "kündigen", " wo "),
        "fr": (" je ", " mon ", " commande", "mot de passe", "annuler", " où "),
        "es": (" mi ", " pedido", "contraseña", "cancelar", " dónde ", " quiero "),
        "bn": ("আমার", "পাসওয়ার্ড", "অর্ডার", "বাতিল", "কোথায়"),
    }
    scores = {code: sum(marker in lowered for marker in values) for code, values in markers.items()}
    code, score = max(scores.items(), key=lambda item: item[1])
    return code if score else "en"


def detect_language(text: str) -> str:
    if not text.strip():
        return "unknown"
    try:
        from langdetect import DetectorFactory, detect

        DetectorFactory.seed = 0
        detected = detect(text)
        return detected if detected in LANGUAGE_NAMES else detected
    except (ImportError, ModuleNotFoundError):
        return _fallback_detect(text)
    except Exception:
        return _fallback_detect(text)


def language_name(code: str) -> str:
    return LANGUAGE_NAMES.get(code, code.upper())
