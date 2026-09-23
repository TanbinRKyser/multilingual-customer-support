from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["knowledge_documents"] >= 3


def test_chat_returns_localized_explainable_answer():
    response = client.post(
        "/chat",
        json={"message": "Ich habe mein Passwort vergessen", "explain_method": "lime"},
    )
    assert response.status_code == 200
    result = response.json()
    assert result["detected_language"] == "de"
    assert result["intent"] == "reset_password"
    assert result["resolution"] == "intent"
    assert result["explanation"]


def test_blank_message_is_rejected():
    response = client.post("/chat", json={"message": "   "})
    assert response.status_code == 422
