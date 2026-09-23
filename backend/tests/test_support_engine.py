from app.services.intent_service import predict_intent
from app.services.knowledge_service import search_knowledge
from app.utils.lang_utils import detect_language


def test_multilingual_intents():
    cases = {
        "I forgot my password": "reset_password",
        "Wo ist meine Bestellung?": "track_order",
        "Je veux annuler ma commande": "cancel_order",
        "Quiero cancelar mi suscripción": "cancel_subscription",
        "I lost my 2FA recovery codes": "recover_2fa",
    }
    for message, expected in cases.items():
        prediction = predict_intent(message)
        assert prediction.intent == expected
        assert prediction.confidence >= .55
        assert prediction.evidence


def test_unknown_intent_is_safe():
    prediction = predict_intent("The weather is nice today")
    assert prediction.intent == "unknown"
    assert prediction.confidence == 0


def test_language_fallbacks():
    assert detect_language("Ich habe mein Passwort vergessen") == "de"
    assert detect_language("Je veux annuler ma commande") == "fr"


def test_knowledge_search_returns_source():
    results = search_knowledge("reset password")
    assert results
    assert results[0][0].source == "password_reset.txt"
