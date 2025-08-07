from fastapi import FastAPI, Request
from pydantic import BaseModel
from app.utils.lang_utils import detect_language
from app.services.chatbot import get_llm_response
from app.services.explain import explain_input_text
from fastapi.middleware.cors import CORSMiddleware
from intent_classifier.bert_infer import predict_intent


app = FastAPI()

# Allow Angular dev server to call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],  # Angular app URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest( BaseModel ):
    message: str
    user_id: str

@app.get("/")
def home():
    return {"message": "Welcome to the mulilingual support chat API!"}

@app.post("/chat")
def chat( request: ChatRequest ):

    lang = detect_language( request.message )
    response = get_llm_response( request.message, language=lang )
    explanation = explain_input_text( request.message )
    intent, confidence = predict_intent(request.message)[0]

    CONF_THRESHOLD = 0.5
    if confidence < CONF_THRESHOLD:
        intent = "Unknown"

    return {
        "response": response,
        "language": lang,
        "explanation": explanation,
        "intent": intent,
        "confidence": round(confidence, 4)
    }