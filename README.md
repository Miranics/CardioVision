# CardioVision

CardioVision is an image-based MLOps pipeline for chest X-ray classification with:

- Single-image prediction API and web UI.
- Uptime and inference metrics endpoints.
- Bulk data upload for retraining.
- Retraining trigger and status tracking.
- Dataset visualization data for UI charts.
- Docker packaging and Locust load testing support.

## Project Structure

```text
CardioVision/
├── README.md
├── notebook/
│   └── MiracleNanenMbanaade_CardioVision.ipynb
├── src/
│   ├── app.py
│   ├── model.py
│   ├── prediction.py
│   └── preprocessing.py
├── templates/
│   └── index.html
├── data/
│   ├── train/
│   ├── val/
│   ├── test/
│   └── uploads/
├── models/
│   ├── cardiovision_model_v1.keras
│   └── cardiovision_model_retrained.keras
├── locustfile.py
├── Dockerfile
└── requirements.txt
```

## Local Setup

1. Create and activate a virtual environment.

```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Run the API/UI app.

```bash
python src/app.py
```

4. Open the app in your browser:

```text
http://127.0.0.1:5000
```

## API Endpoints

- `GET /health`
	- Returns API health, uptime, active model path, and retrain state.

- `GET /metrics`
	- Returns total predictions, average latency, and last prediction timestamp.

- `GET /visualization-data`
	- Returns class labels and counts from `data/train` for UI visualization.

- `GET /data-status`
	- Returns dataset split existence and per-class counts for `train`, `val`, and `test`.

- `POST /predict`
	- Form-data: `file` (single image)
	- Returns class prediction, confidence, probability, and latency.

- `POST /upload-retrain-data`
	- Form-data: `class_label` (`NORMAL` or `PNEUMONIA`), `files` (multiple images)
	- Saves uploaded images to `data/uploads/<CLASS_LABEL>/`.

- `POST /trigger-retrain`
	- JSON: `{ "epochs": 3 }`
	- Starts background retraining.

- `GET /retrain-status`
	- Returns current retraining status, timestamps, and latest result.

## Retraining Workflow

1. Upload new images with `POST /upload-retrain-data`.
2. Trigger retraining with `POST /trigger-retrain`.
3. Poll `GET /retrain-status` until state is `completed` or `failed`.
4. On success, the app switches to the new retrained model path.

## Run with Docker

Build image:

```bash
docker build -t cardiovision:latest .
```

Run container:

```bash
docker run --rm -p 5000:5000 cardiovision:latest
```

Or with Docker Compose:

```bash
docker compose up --build
```

## Flood Testing with Locust

1. Set a sample image path for prediction traffic.

```bash
export LOCUST_SAMPLE_IMAGE="/absolute/path/to/sample_xray.jpg"
```

2. Start app and Locust:

```bash
locust -f locustfile.py --host=http://127.0.0.1:5000
```

Headless run helper:

```bash
chmod +x scripts/run_locust_headless.sh
./scripts/run_locust_headless.sh /absolute/path/to/sample_xray.jpg 50 10 3m http://127.0.0.1:5000
```

You can also omit the sample image argument and the script will auto-pick one from `data/test` or `data/val`.

3. Open Locust UI:

```text
http://127.0.0.1:8089
```

## Quick Smoke Test

After running the API, execute:

```bash
./scripts/smoke_test_api.sh
```

This verifies `/health`, `/metrics`, `/visualization-data`, and `/data-status`.

Dataset readiness check:

```bash
"./venv/bin/python" scripts/check_dataset.py
```

This fails fast when required split/class folders are empty.

## Training Report Script

Run a repeatable training pass and save metrics to `reports/`:

```bash
CV_TRAIN_EPOCHS=3 CV_TRAIN_LR=1e-4 "./venv/bin/python" scripts/train_and_report.py
```

## Deploy: Render Backend + Vercel Frontend

### 1. Deploy backend to Render (Docker)

This repo includes `render.yaml` configured for Docker runtime and `/health` checks.

On Render:

1. Create a new Web Service from this GitHub repo.
2. Render will detect `render.yaml` automatically.
3. After first deploy, copy your backend URL, for example:

```text
https://cardiovision-api.onrender.com
```

4. In Render environment variables, set:

```text
CORS_ALLOWED_ORIGINS=https://your-vercel-project.vercel.app
```

If you use a custom frontend domain, include it too as comma-separated values.

### 2. Deploy frontend to Vercel

Frontend files are in `frontend/`.

On Vercel:

1. Import this GitHub repo.
2. Set Root Directory to `frontend`.
3. Deploy.
4. Open the Vercel app URL, paste your Render backend URL in the UI, and click **Save Backend URL**.

The frontend stores your backend URL in browser local storage and calls Render API endpoints directly.

## Notebook

The notebook includes model experimentation, preprocessing, and evaluation metrics. This implementation keeps notebook content unchanged and implements deployment/runtime operations in `src/`.

