from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from intent_classifier.bert_infer import predict_intent
from app.utils.lang_utils import detect_language
from app.services.rag_service import query_rag  

CONF_THRESHOLD = 0.5

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    original_message: str
    detected_language: str
    intent: str
    confidence: float
    response: str
    sources: list | None = None

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4300", "http://127.0.0.1:4300"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "Welcome to the multilingual support chat API!"}

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    message = request.message

    # Step 1: Detect language
    try:
        lang = detect_language(message)
    except Exception:
        lang = "unknown"

    # Step 2: Predict intent with BERT
    try:
        intent, confidence = predict_intent(message)[0]
    except Exception:
        intent, confidence = "Unknown", 0.0

    # Step 3: Mock responses for high-confidence matches
    mock_responses = {
        "reset_password": "To reset your password, click 'Forgot Password' on the login page.",
        "track_order": "Track your order in the 'My Orders' section. Share your order ID if you need help.",
        "cancel_order": "To cancel your order, open 'My Orders', select the item, and choose 'Cancel'.",
        "channel_subscription": "Go to 'My Subscriptions' and follow the cancellation steps.",
        "default": "I'm not sure—let me check our knowledge base."
    }

    # Decision: if high confidence & known intent → mock response
    if confidence >= CONF_THRESHOLD and intent in mock_responses:
        response_text = mock_responses[intent]
        sources = None
    else:
        # Fallback: RAG retrieval + FLAN-T5 generation
        rag_result = query_rag(message, top_k=3)
        response_text = rag_result["answer"]
        sources = rag_result["sources"]

    return ChatResponse(
        original_message=message,
        detected_language=lang,
        intent=intent,
        confidence=round(confidence, 4),
        response=response_text,
        sources=sources
    )
