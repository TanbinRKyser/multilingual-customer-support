from fastapi import FastAPI, Request
from pydantic import BaseModel


app = FastAPI()

class ChatRequest(BaseModel):
    message: str
    user_id: str

@app.get("/")
def home():
    return {"message": "Welcome to the mulilingual support chat API!"}

@app.post("/chat")
def chat(request: ChatRequest):
    return {
        "response": f"You said: {request.message}",
        "language": "en",
    }