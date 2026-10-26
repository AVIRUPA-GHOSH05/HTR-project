from __future__ import annotations

from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from app.model_runner import ModelRunner

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_TYPES = {"image/png", "image/jpeg", "image/bmp", "image/x-ms-bmp"}
ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}
runner = ModelRunner()
BASE_DIR = Path(__file__).resolve().parent


@asynccontextmanager
async def lifespan(_: FastAPI):
    runner.load()
    yield


app = FastAPI(title="Handwriting AI", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


class PDFRequest(BaseModel):
    text: str


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return (BASE_DIR / "static" / "index.html").read_text(encoding="utf-8")


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", "model_loaded": runner.loaded, "detail": runner.reason}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, str]:
    suffix = Path(file.filename or "").suffix.lower()
    if file.content_type not in ALLOWED_TYPES or suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=400, detail="Upload a PNG, JPEG, or BMP image.")
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 10 MB upload limit.")
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")
    try:
        text = runner.predict(content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Image recognition failed.") from exc
    return {"text": text}


@app.post("/download-pdf")
async def download_pdf(request: PDFRequest) -> StreamingResponse:
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    page_width, page_height = letter
    left, top, line_height = 54, page_height - 60, 16
    pdf.setTitle("Handwriting transcription")
    pdf.setFont("Helvetica", 11)
    y = top
    for paragraph in request.text.splitlines() or [""]:
        words = paragraph.split()
        line = ""
        for word in words:
            candidate = f"{line} {word}".strip()
            if pdf.stringWidth(candidate, "Helvetica", 11) > page_width - 2 * left and line:
                if y < 54:
                    pdf.showPage()
                    pdf.setFont("Helvetica", 11)
                    y = top
                pdf.drawString(left, y, line)
                y -= line_height
                line = word
            else:
                line = candidate
        if y < 54:
            pdf.showPage()
            pdf.setFont("Helvetica", 11)
            y = top
        pdf.drawString(left, y, line)
        y -= line_height
    pdf.save()
    buffer.seek(0)
    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="transcription.pdf"'},
    )
