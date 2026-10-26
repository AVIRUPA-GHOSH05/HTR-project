from fastapi.testclient import TestClient

from app.main import app, runner


def test_health_reports_model_state():
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert "model_loaded" in response.json()


def test_predict_rejects_non_image_upload():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            files={"file": ("notes.txt", b"not an image", "text/plain")},
        )
    assert response.status_code == 400


def test_predict_reports_missing_weights(monkeypatch):
    monkeypatch.setattr(runner, "predict", lambda _: (_ for _ in ()).throw(RuntimeError("No TensorFlow checkpoint found in pretrained/.")))
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            files={"file": ("line.png", b"png bytes", "image/png")},
        )
    assert response.status_code == 503
    assert "checkpoint" in response.json()["detail"].lower()


def test_pdf_endpoint_returns_pdf():
    with TestClient(app) as client:
        response = client.post("/download-pdf", json={"text": "This is a transcription."})
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.content.startswith(b"%PDF")


def test_pdf_handles_long_text_across_pages():
    text = "A handwritten sentence for the PDF. " * 1000
    with TestClient(app) as client:
        response = client.post("/download-pdf", json={"text": text})
    assert response.status_code == 200
    assert response.content.startswith(b"%PDF")
