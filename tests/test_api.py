from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "rag-evaluation-observability-lab",
    }


def test_query_validation_rejects_empty_question() -> None:
    response = client.post(
        "/query",
        json={
            "question": "",
            "top_k": 3,
        },
    )

    assert response.status_code == 422