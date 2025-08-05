from fastapi import FastAPI, Request
from pydantic import BaseModel
from app.utils.lang_utils import detect_language

app = FastAPI()

class ChatRequest(BaseModel):
    message: str
    user_id: str

@app.get("/")
def home():
    return {"message": "Welcome to the mulilingual support chat API!"}

@app.post("/chat")
def chat(request: ChatRequest):
    lang = detect_language(request.message)
    return {
        "response": f"You said: {request.message}",
        "language": lang,
    }