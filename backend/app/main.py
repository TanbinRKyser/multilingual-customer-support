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
    # user_id: str

class ChatResponse( BaseModel ):
    original_message: str
    detected_language: str
    intent: str
    confidence: float
    response: str
    # explanation: list[dict[str, float]] = None

@app.get("/")
def home():
    return {"message": "Welcome to the multilingual support chat API!"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint( request: ChatRequest ):

    message = request.message


    ## step 1 : Detect language
    try: lang = detect_language( message )
    except Exception as e:
        lang = "unknown"
    
    ## step 2 : Predict intent using BERT
    try: intent, confidence = predict_intent( message )[0]
    except Exception as e:
        intent = "Unknown"
        confidence = 0.0
        print(f"Intent prediction failed: {e}")

    ## step 3 : Generate Mock Response
    mock_responses = {
        "reset_password": "To reset your password...",
        "track_order": "You can track your order...",
        "cancel_order": "To cancel your order, go to 'My Orders' and select 'Cancel'.",
        "channel_subscription": "...",
        "default": "I'm sorry, I didn't understand that...",
    }


    CONF_THRESHOLD = 0.35

    if confidence < CONF_THRESHOLD or intent == "Unknown":
        # Low confidence or no match — fallback to default message
        response_text = mock_responses["default"]
    elif intent in mock_responses:
        # Confident prediction and we have a response for it
        response_text = mock_responses[intent]
    else:
        # Intent predicted but not in mock_responses — safe fallback
        response_text = mock_responses["default"]

    ## step 4 : Generate explanation
    # explanation = explain_input_text( request.message )


    # lang = detect_language( request.message )
    # response = get_llm_response( request.message, language=lang )
    # explanation = explain_input_text( request.message )
    # intent, confidence = predict_intent(request.message)[0]

    # CONF_THRESHOLD = 0.5
    # if confidence < CONF_THRESHOLD:
    #     intent = "Unknown"

    # return {
    #     "response": response,
    #     "language": lang,
    #     "explanation": explanation,
    #     "intent": intent,
    #     "confidence": round(confidence, 4)
    # }

    return ChatResponse(
        original_message=message,
        detected_language=lang,
        intent=intent,
        confidence=round(confidence, 4),
        response=response_text,
        # explanation=explanation
    )