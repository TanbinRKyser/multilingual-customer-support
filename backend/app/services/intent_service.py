"""Small, deterministic multilingual intent router.

The repository's fine-tuned BERT artifacts are optional. This classifier gives the
application a useful, explainable baseline even when those large artifacts are not
present (which is also ideal for tests and a first Docker run).
"""

from dataclasses import dataclass

INTENT_TERMS: dict[str, dict[str, float]] = {
    "reset_password": {
        "password": 1.0, "passwort": 1.0, "passe": .65, "contraseña": 1.0,
        "পাসওয়ার্ড": 1.0, "forgot": .75, "vergessen": .75, "oublié": .75,
        "reset": .7, "zurücksetzen": .7, "recovery": .55, "login": .45,
    },
    "track_order": {
        "track": 1.0, "tracking": 1.0, "shipment": .8, "package": .7,
        "order": .45, "bestellung": .6, "verfolgen": 1.0, "commande": .55,
        "suivre": 1.0, "pedido": .55, "rastrear": 1.0, "অর্ডার": .6, "কোথায়": .7,
    },
    "cancel_order": {
        "cancel": 1.0, "stornieren": 1.0, "annuler": 1.0, "cancelar": 1.0,
        "বাতিল": 1.0, "order": .35, "bestellung": .4, "commande": .4, "pedido": .4,
    },
    "cancel_subscription": {
        "subscription": .8, "membership": .8, "abonnement": .8, "suscripción": .8,
        "cancel": .8, "kündigen": 1.0, "annuler": .8, "cancelar": .8,
    },
    "recover_2fa": {
        "2fa": 1.0, "two-factor": 1.0, "authenticator": .8, "recovery": .7,
        "code": .45, "codes": .45, "key": .35, "schlüssel": .45,
    },
}


@dataclass(frozen=True)
class IntentPrediction:
    intent: str
    confidence: float
    evidence: list[dict[str, float | str]]


def predict_intent(text: str) -> IntentPrediction:
    lowered = text.casefold()
    scored: list[tuple[str, float, list[dict[str, float | str]]]] = []
    for intent, vocabulary in INTENT_TERMS.items():
        evidence = [
            {"token": term, "weight": weight}
            for term, weight in vocabulary.items()
            # Substring matching also handles scripts whose combining marks are
            # split by Python's generic word-boundary regular expressions.
            if term in lowered
        ]
        scored.append((intent, sum(float(item["weight"]) for item in evidence), evidence))

    intent, score, evidence = max(scored, key=lambda item: item[1])
    if score == 0:
        return IntentPrediction("unknown", 0.0, [])

    # A single strong keyword is useful, while two independent matches are decisive.
    confidence = min(.98, .42 + (score * .24) + (max(0, len(evidence) - 1) * .08))
    max_weight = max(float(item["weight"]) for item in evidence)
    normalized = [
        {"token": str(item["token"]), "weight": round(float(item["weight"]) / max_weight, 3)}
        for item in evidence
    ]
    return IntentPrediction(intent, round(confidence, 4), normalized)
