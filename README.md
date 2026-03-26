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

## Notebook

The notebook includes model experimentation, preprocessing, and evaluation metrics. This implementation keeps notebook content unchanged and implements deployment/runtime operations in `src/`.

## Next Submission Items to Add

- Public deployment URL.
- YouTube demo link.
- Captured Locust result tables/screenshots (latency and response time by load and container count).
