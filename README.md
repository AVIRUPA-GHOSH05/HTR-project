# Handwriting AI

A responsive handwriting line recognition web app built with HTML, CSS, vanilla JavaScript, FastAPI, and the repository's original TensorFlow CNN–BiLSTM–CTC model.

## Tech stack and Python packages

- **Backend/API:** Python 3.10 and FastAPI, served by Uvicorn.
- **Handwriting recognition:** TensorFlow 2.12 using its `compat.v1` API, with the original CNN–BiLSTM–CTC model.
- **Image handling:** OpenCV and NumPy for decoding and preprocessing uploaded line images.
- **Frontend:** HTML, CSS, and vanilla JavaScript, served by FastAPI. No Node.js or frontend package manager is needed.
- **PDF export:** ReportLab.
- **Tests:** pytest and HTTPX.

Install the Python packages from the project root with `pip install -r requirements.txt`. The file pins TensorFlow and constrains NumPy for compatibility; it also installs:

| Package | Purpose |
| --- | --- |
| `tensorflow==2.12.0` | Loads the trained recognition model. |
| `numpy` | Image arrays and model input. |
| `opencv-python-headless` | Reads and preprocesses uploaded images without a desktop GUI dependency. |
| `fastapi` | Web API and static frontend server. |
| `uvicorn[standard]` | Runs the FastAPI application. |
| `python-multipart` | Parses image uploads sent as multipart form data. |
| `reportlab` | Generates transcription PDFs. |
| `pytest` and `httpx` | Run the API tests. |

No separate frontend modules are required. The model checkpoint is also not a pip package: it must be obtained separately and placed in `pretrained/` as described below. Checkpoint files are excluded from the repository.

## Model and weights

The original model implementation and preprocessing code are preserved in `src/Model.py`, `src/SamplePreprocessor.py`, and `src/DataLoader.py`. Its character vocabulary is in `data/charList.txt`. The model expects grayscale line images resized and padded to 800×64, then transposed, and decodes with TensorFlow's greedy CTC decoder.

The trained checkpoint is not included in the repository. The app does not fabricate predictions: `/predict` returns HTTP 503 until a compatible TensorFlow checkpoint is installed. Download a checkpoint produced by the original architecture, then place its files and TensorFlow checkpoint state file under `pretrained/` (for example `pretrained/snapshot-24.index`, its matching `.data-*` shard, and a `pretrained/checkpoint` file that points to it). Keep it compatible with `data/charList.txt`. The model loader reports unavailable if it cannot find a restorable checkpoint.

The TensorFlow 2 training/inference scripts in the upstream source are not used: the original TF1-compatible graph model is the closest complete inference implementation, and the local TF2 inference script has unresolved references. Runtime compatibility is targeted at Python 3.10 with TensorFlow 2.12's `compat.v1` API. Training is not part of this web app. To train from scratch also requires the IAM line dataset (`data/lines.txt` and `data/lines/...`), which was not supplied.

## Features

- Drag-and-drop or file-picker upload for PNG, JPEG, and BMP images, preview, and client/server validation.
- Real model inference through `POST /predict` once checkpoint files are present.
- Editable transcription, copy, clear, and PDF download controls.
- `GET /health` for model availability and `POST /download-pdf` for ReportLab PDF generation.
- API and PDF tests, including the missing-weights response.

## Windows PowerShell setup

Use Python 3.10 (TensorFlow 2.12 supports this runtime). From the project root:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn app.main:app --reload
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process Bypass` in that terminal and activate again.

## Linux/macOS setup

Install Python 3.12, then from the project root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python --version
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn app.main:app --reload
```

In VS Code, select `.venv` with **Python: Select Interpreter** if imports still appear unresolved. The workspace default points to `.venv`, but VS Code keeps an interpreter already selected for the workspace until you change it.

Open <http://127.0.0.1:8000>. API docs are at <http://127.0.0.1:8000/docs>. TensorFlow wheel support varies by operating system and hardware; install the matching TensorFlow 2.12 wheel if the pinned package cannot be installed on your platform.

## API

- `GET /health` → `{"status":"ok","model_loaded":false,...}`. The service remains available when weights are missing so the UI and PDF feature can be used.
- `POST /predict` → multipart form field `file`; success returns `{"text":"..."}`. Returns 400 for unsupported/corrupt images, 413 for oversized uploads, and 503 when a checkpoint or model runtime is unavailable.
- `POST /download-pdf` → JSON `{"text":"..."}`; returns an `application/pdf` download.

The model accepts line images. A full page or paragraph may need line segmentation for best results.

## Tests

```bash
pytest -q
```

Tests do not require model weights. Actual recognition is not verifiable until compatible checkpoint files are supplied.
