# CardioVision


Cardiovascular diseases are a leading cause of death worldwide. Early detection is crucial for better outcomes. This project uses deep learning to classify chest X-ray images as Normal or Abnormal, aiming to detect potential cardiovascular risks.
CardioVision is an end-to-end machine learning classification project for chest X-ray images. It covers:

1. Offline model development and evaluation in Jupyter.
2. API and web UI for prediction and retraining workflows.
3. Docker packaging and cloud deployment.
4. Request flood simulation with Locust.

## Live Links
1. Demo video: https://youtu.be/raBeCyfqjfU?si=9E_pcpx0SoUOR7EP
2. Frontend URL: https://cardiovision-api-2lza.onrender.com
3. Backend URL: https://cardiovision-api-2lza.onrender.com
4. Health check: https://cardiovision-api-2lza.onrender.com/health


## Project Objective

Demonstrate the full Machine Learning lifecycle for a non-tabular dataset (images), including:

1. Data acquisition and preprocessing.
2. Model training and testing.
3. Retraining trigger and automation.
4. Production API and UI.
5. Monitoring and flood testing.

## Repository Structure

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
├── scripts/
│   ├── check_dataset.py
│   ├── run_locust_headless.sh
│   ├── smoke_test_api.sh
│   └── train_and_report.py
├── frontend/
│   ├── index.html
│   └── assets/
├── data/
│   ├── train/
│   ├── val/
│   ├── test/
│   └── uploads/
├── models/
│   ├── cardiovision_model_v1.keras
│   └── cardiovision_model_retrained.keras
├── reports/
├── Dockerfile
├── docker-compose.yml
├── locustfile.py
└── requirements.txt
```

## What an Assessor Should Test First

1. Open the live frontend.
2. Upload one image and run prediction.
3. Upload multiple images for retraining.
4. Trigger retraining and monitor status.
5. Check uptime and metrics are updating.

## Web App Guide (Every Input Explained)

### Section 1: X-ray Prediction

1. Input: file picker under X-ray Prediction.
2. Accepted input: one image file.
3. Action button: Run Clinical Prediction.
4. Output:
	1. Predicted class label.
	2. Confidence value.
	3. Inference latency in milliseconds.

### Section 2: Retraining Intake

1. Input: class selector.
	1. NORMAL
	2. PNEUMONIA
2. Input: multiple image upload field.
3. Action button: Upload New Dataset Images.
4. Behavior:
	1. Files are stored under data/uploads per class.
	2. Upload confirmation message appears on success.

### Section 3: Model Retraining Control

1. Input: epochs number field.
2. Action button: Trigger Retraining.
3. Output:
	1. Current retraining state (idle, running, completed, failed).
	2. Status message for progress or error.

### Section 4: Monitoring and Insights

1. System Health: backend availability status.
2. API Uptime: running time of backend process.
3. Predictions: cumulative inference count.
4. Avg Latency: average inference duration.
5. Dataset insights:
	1. Train volume.
	2. Class balance score.
	3. Split readiness summary.

## Local Setup (Python)

1. Create and activate virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the backend app:

```bash
python src/app.py
```

4. Open local URL:

```text
http://127.0.0.1:5000
```

## Docker Setup

1. Build image:

```bash
docker build -t cardiovision:latest .
```

2. Run container:

```bash
docker run --rm -p 5000:5000 cardiovision:latest
```

3. Or run with Compose:

```bash
docker compose up --build
```

## API Endpoints

1. GET /health: status, uptime, active model, retrain state.
2. GET /metrics: prediction count, avg latency, last prediction time.
3. GET /visualization-data: labels and counts for training visualization.
4. GET /data-status: split existence and class counts.
5. POST /predict: predict one uploaded image.
6. POST /upload-retrain-data: upload multiple labeled images.
7. POST /trigger-retrain: start retraining with epochs value.
8. GET /retrain-status: monitor retraining state and result.

## Retraining Flow

1. Upload multiple labeled images in Retraining Intake.
2. Set epoch value in Model Retraining Control.
3. Trigger retraining.
4. Track state until completed.
5. New model becomes active after successful retrain.

## Flood Testing (Locust)

### Option A: Locust UI

1. Start backend first.
2. Run Locust:

```bash
locust -f locustfile.py --host=http://127.0.0.1:5000
```

3. Open Locust UI:

```text
http://127.0.0.1:8089
```

### Option B: Headless Script

```bash
chmod +x scripts/run_locust_headless.sh
./scripts/run_locust_headless.sh /absolute/path/to/sample_xray.jpg 50 10 3m http://127.0.0.1:5000
```

## Flood Testing Results (Fill Before Submission)

Measured results from headless Locust runs (local Docker backend).

Test conditions:

1. Target: `http://127.0.0.1:5000`
2. Container count: 1 (Docker Compose single service instance)
3. Script used: `scripts/run_locust_headless.sh`
4. Sample image: auto-picked from `data/test/NORMAL`
5. Request mix from `locustfile.py`: health (weight 1), predict (weight 3)

| Containers | Users | Spawn Rate | Duration | Avg Response (ms) | P95 (ms) | Failure Rate (%) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 20 | 5/s | 45s | 1550 | 2000 | 0.00 |
| 1 | 40 | 10/s | 45s | 3363 | 4100 | 0.00 |
| 1 | 60 | 15/s | 45s | 4830 | 6100 | 0.00 |

Interpretation notes:

1. Latency increased as concurrent users increased, which is expected under higher load.
2. Failure rate remained 0% across all three scenarios.
3. For this machine and model size, a practical baseline is to keep user load near the first scenario range unless horizontal scaling is added.

Exact commands used:

```bash
./scripts/run_locust_headless.sh '' 20 5 45s http://127.0.0.1:5000
./scripts/run_locust_headless.sh '' 40 10 45s http://127.0.0.1:5000
./scripts/run_locust_headless.sh '' 60 15 45s http://127.0.0.1:5000
```

## Utility Scripts

1. Smoke test API:

```bash
./scripts/smoke_test_api.sh
```

2. Dataset readiness check:

```bash
./venv/bin/python scripts/check_dataset.py
```

3. Training report generation:

```bash
CV_TRAIN_EPOCHS=3 CV_TRAIN_LR=1e-4 ./venv/bin/python scripts/train_and_report.py
```

## Deployment Notes

### Backend on Render (Docker)

1. Runtime: Docker.
2. Dockerfile path: Dockerfile.
3. Health check path: /health.
4. Environment variables:
	1. PYTHONUNBUFFERED=1
	2. CORS_ALLOWED_ORIGINS=https://frontend-mu-eight-31.vercel.app

### Frontend on Vercel

1. Root directory: frontend.
2. Current frontend is already configured to use the deployed backend URL.

## Notebook and Evaluation


1. File: notebook/MiracleNanenMbanaade_CardioVision.ipynb
2. Contains preprocessing, model training, and evaluation content.
3. Includes required model quality metrics and experimentation details.

## Conclusion

CardioVision demonstrates a complete machine learning lifecycle for medical image classification: from data preparation and model development to deployment, monitoring, and retraining.


