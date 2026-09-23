from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.models.schemas import ChatRequest, ChatResponse, HealthResponse, Source
from app.services.intent_service import predict_intent
from app.services.knowledge_service import load_documents, search_knowledge
from app.utils.lang_utils import detect_language, language_name


INTENT_QUERIES = {
    "reset_password": "reset forgot password login",
    "track_order": "track order shipment package",
    "cancel_order": "cancel order",
    "cancel_subscription": "cancel subscription membership",
    "recover_2fa": "2FA recovery codes authentication",
}

RESPONSES = {
    "en": {
        "reset_password": "Select “Forgot Password” on the sign-in page, enter your registered email, and use the link we send you. The link expires after 24 hours.",
        "track_order": "Sign in, open My Orders, and select Track next to the order. If the tracking link fails, contact support with your order ID.",
        "cancel_order": "Open My Orders, select the order, and choose Cancel. If that option is no longer available, the order may already be processing and an agent can help.",
        "cancel_subscription": "Open My Subscriptions, choose the active plan, and follow the cancellation steps. Your access continues through the current billing period.",
        "recover_2fa": "If you lost your two-factor recovery codes, contact your IT administrator or support team so they can verify your identity and restore access.",
        "handoff": "I couldn’t confidently match that request. Please rephrase it, or ask for a support agent.",
    },
    "de": {
        "reset_password": "Wählen Sie auf der Anmeldeseite „Passwort vergessen“, geben Sie Ihre registrierte E-Mail-Adresse ein und öffnen Sie den zugesandten Link. Er ist 24 Stunden gültig.",
        "track_order": "Melden Sie sich an, öffnen Sie „Meine Bestellungen“ und wählen Sie neben der Bestellung „Verfolgen“. Bei Problemen hilft der Support mit Ihrer Bestellnummer.",
        "cancel_order": "Öffnen Sie „Meine Bestellungen“, wählen Sie die Bestellung und dann „Stornieren“. Fehlt diese Option, kann der Support den Status prüfen.",
        "cancel_subscription": "Öffnen Sie „Meine Abonnements“, wählen Sie den aktiven Tarif und folgen Sie den Schritten zur Kündigung.",
        "recover_2fa": "Wenn Ihre Wiederherstellungscodes fehlen, wenden Sie sich zur Identitätsprüfung und Wiederherstellung des Zugangs an Ihre IT-Administration oder den Support.",
        "handoff": "Ich konnte die Anfrage nicht sicher zuordnen. Formulieren Sie sie bitte anders oder bitten Sie um einen Support-Mitarbeiter.",
    },
    "fr": {
        "reset_password": "Sélectionnez « Mot de passe oublié » sur la page de connexion, saisissez votre adresse e-mail et ouvrez le lien reçu. Il expire après 24 heures.",
        "track_order": "Connectez-vous, ouvrez « Mes commandes », puis sélectionnez « Suivre ». Si le lien échoue, contactez l’assistance avec votre numéro de commande.",
        "cancel_order": "Ouvrez « Mes commandes », sélectionnez la commande, puis « Annuler ». Si l’option n’apparaît plus, contactez l’assistance.",
        "cancel_subscription": "Ouvrez « Mes abonnements », sélectionnez l’offre active et suivez les étapes de résiliation.",
        "recover_2fa": "Si vous avez perdu vos codes de récupération, contactez votre administrateur informatique ou l’assistance pour vérifier votre identité.",
        "handoff": "Je n’ai pas pu identifier votre demande avec certitude. Reformulez-la ou demandez un conseiller.",
    },
    "es": {
        "reset_password": "Selecciona «Olvidé mi contraseña» en la página de acceso, introduce tu correo registrado y abre el enlace recibido. Caduca en 24 horas.",
        "track_order": "Inicia sesión, abre «Mis pedidos» y selecciona «Rastrear». Si el enlace falla, contacta con soporte e indica tu número de pedido.",
        "cancel_order": "Abre «Mis pedidos», selecciona el pedido y elige «Cancelar». Si la opción ya no aparece, contacta con soporte.",
        "cancel_subscription": "Abre «Mis suscripciones», selecciona el plan activo y sigue los pasos de cancelación.",
        "recover_2fa": "Si perdiste los códigos de recuperación, contacta con tu administrador o con soporte para verificar tu identidad.",
        "handoff": "No pude identificar la solicitud con suficiente certeza. Reformúlala o pide hablar con un agente.",
    },
    "bn": {
        "reset_password": "সাইন-ইন পেজে ‘পাসওয়ার্ড ভুলে গেছেন’ নির্বাচন করুন, নিবন্ধিত ইমেইল দিন এবং পাঠানো লিংকটি খুলুন। লিংকটি ২৪ ঘণ্টা কার্যকর থাকবে।",
        "track_order": "সাইন ইন করে ‘আমার অর্ডার’ খুলুন এবং অর্ডারের পাশে ‘ট্র্যাক’ নির্বাচন করুন। সমস্যা হলে অর্ডার নম্বরসহ সাপোর্টে যোগাযোগ করুন।",
        "cancel_order": "‘আমার অর্ডার’ খুলে অর্ডারটি নির্বাচন করুন, তারপর ‘বাতিল’ চাপুন। অপশনটি না থাকলে সাপোর্টে যোগাযোগ করুন।",
        "cancel_subscription": "‘আমার সাবস্ক্রিপশন’ খুলে সক্রিয় প্ল্যান নির্বাচন করুন এবং বাতিল করার ধাপগুলো অনুসরণ করুন।",
        "recover_2fa": "রিকভারি কোড হারালে পরিচয় যাচাই ও অ্যাক্সেস ফেরত পেতে আইটি অ্যাডমিন বা সাপোর্টে যোগাযোগ করুন।",
        "handoff": "আমি অনুরোধটি নিশ্চিতভাবে বুঝতে পারিনি। অন্যভাবে লিখুন অথবা একজন সাপোর্ট এজেন্ট চান।",
    },
}

app = FastAPI(
    title="Polyglot Support API",
    description="Explainable multilingual intent routing with knowledge-base grounding.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "http://127.0.0.1:4200", "http://localhost:4300", "http://127.0.0.1:4300"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {"message": "Polyglot Support API", "docs": "/docs", "health": "/health"}


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(knowledge_documents=len(load_documents()))


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    message = request.message
    language = detect_language(message)
    prediction = predict_intent(message)
    locale = language if language in RESPONSES else "en"

    if prediction.confidence >= .55 and prediction.intent in INTENT_QUERIES:
        intent = prediction.intent
        response_text = RESPONSES[locale][intent]
        matches = search_knowledge(INTENT_QUERIES[intent], limit=2)
        resolution = "intent"
    else:
        matches = search_knowledge(message, limit=1)
        intent = "unknown"
        if matches and locale == "en":
            response_text = matches[0][0].text
            resolution = "knowledge_base"
        else:
            response_text = RESPONSES[locale]["handoff"]
            resolution = "handoff"

    sources = [
        Source(source=doc.source, snippet=doc.text[:180] + ("…" if len(doc.text) > 180 else ""))
        for doc, _score in matches
    ]
    explanation = prediction.evidence if request.explain_method else []

    return ChatResponse(
        original_message=message,
        detected_language=language,
        language_name=language_name(language),
        intent=intent,
        confidence=prediction.confidence,
        response=response_text,
        resolution=resolution,
        sources=sources,
        explanation=explanation,
    )
