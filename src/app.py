"""Flask API service for CardioVision prediction, monitoring, and retraining."""

import os
import json
import threading
import time
from datetime import datetime

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

from prediction import predict_from_uploaded_file, set_model_path
from preprocessing import (
    CLASS_NAMES,
    count_images_by_class,
    dataset_split_status,
    save_uploaded_files,
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
REPORTS_DIR = os.path.join(ROOT_DIR, "reports")
MODEL_V1 = os.path.join(MODELS_DIR, "cardiovision_model_v1.keras")
MODEL_RETRAINED = os.path.join(MODELS_DIR, "cardiovision_model_retrained.keras")

ACTIVE_MODEL_PATH = MODEL_RETRAINED if os.path.exists(MODEL_RETRAINED) else MODEL_V1
if not os.path.exists(ACTIVE_MODEL_PATH):
    raise FileNotFoundError(
        "No model found. Expected cardiovision_model_v1.keras or cardiovision_model_retrained.keras in models/."
    )

set_model_path(ACTIVE_MODEL_PATH)

app = Flask(
    __name__,
    template_folder=os.path.join(ROOT_DIR, "templates"),
)
app.config["MAX_CONTENT_LENGTH"] = int(
    os.getenv("MAX_CONTENT_LENGTH_MB", "8")
) * 1024 * 1024

cors_allowed_origins = os.getenv("CORS_ALLOWED_ORIGINS", "*").strip()
if cors_allowed_origins == "*":
    CORS(app)
else:
    allowed = [origin.strip() for origin in cors_allowed_origins.split(",") if origin.strip()]
    CORS(app, origins=allowed)

METRICS = {
    "total_predictions": 0,
    "total_latency_ms": 0.0,
    "last_prediction_at": None,
}

RETRAIN_STATUS = {
    "state": "idle",
    "started_at": None,
    "finished_at": None,
    "message": "No retraining started yet.",
    "last_result": None,
}

STATUS_LOCK = threading.Lock()
APP_START_TIME = time.time()


def _update_retrain_status(**kwargs):
    """Update shared retraining status atomically."""
    with STATUS_LOCK:
        RETRAIN_STATUS.update(kwargs)


def _write_retrain_report(result, epochs):
    """Persist a JSON report for each UI-triggered retraining run."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    report_path = os.path.join(REPORTS_DIR, f"retraining_report_{timestamp}.json")

    payload = {
        "timestamp_utc": datetime.utcnow().isoformat(),
        "trigger": "ui_retrain",
        "epochs": epochs,
        "result": result,
    }

    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    return report_path


def _retrain_worker(epochs):
    """Run retraining asynchronously and update active model when successful."""
    global ACTIVE_MODEL_PATH
    _update_retrain_status(
        state="running",
        started_at=datetime.utcnow().isoformat(),
        finished_at=None,
        message="Retraining in progress.",
    )

    try:
        from model import retrain_from_uploaded_data

        result = retrain_from_uploaded_data(
            base_data_dir=DATA_DIR,
            uploads_dir=UPLOADS_DIR,
            models_dir=MODELS_DIR,
            epochs=epochs,
            batch_size=max(4, int(os.getenv("UI_RETRAIN_BATCH_SIZE", "8"))),
        )

        report_path = _write_retrain_report(result=result, epochs=epochs)
        result["report_path"] = report_path

        ACTIVE_MODEL_PATH = result["saved_model_path"]
        set_model_path(result["saved_model_path"])
        _update_retrain_status(
            state="completed",
            finished_at=datetime.utcnow().isoformat(),
            message="Retraining completed successfully.",
            last_result=result,
        )
    except Exception as exc:
        _update_retrain_status(
            state="failed",
            finished_at=datetime.utcnow().isoformat(),
            message=str(exc),
        )


@app.route("/")
def home():
    """Render the local HTML interface."""
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    """Return service health, uptime, and active model metadata."""
    uptime_seconds = int(time.time() - APP_START_TIME)
    with STATUS_LOCK:
        retrain_state = RETRAIN_STATUS["state"]

    return jsonify(
        {
            "status": "ok",
            "uptime_seconds": uptime_seconds,
            "active_model_path": ACTIVE_MODEL_PATH,
            "retrain_state": retrain_state,
        }
    )


@app.route("/metrics", methods=["GET"])
def metrics():
    """Expose runtime inference metrics for monitoring."""
    total_predictions = METRICS["total_predictions"]
    avg_latency = (
        METRICS["total_latency_ms"] / total_predictions
        if total_predictions > 0
        else 0.0
    )
    return jsonify(
        {
            "total_predictions": total_predictions,
            "average_latency_ms": round(avg_latency, 2),
            "last_prediction_at": METRICS["last_prediction_at"],
        }
    )


@app.route("/visualization-data", methods=["GET"])
def visualization_data():
    """Return train split class counts for UI visualizations."""
    counts = count_images_by_class(TRAIN_DIR)
    labels = CLASS_NAMES
    values = [counts[label] for label in labels]
    return jsonify({"labels": labels, "counts": values})


@app.route("/data-status", methods=["GET"])
def data_status():
    """Return dataset split readiness details across train/val/test."""
    status = dataset_split_status(DATA_DIR)
    return jsonify(status)


@app.route("/predict", methods=["POST"])
def predict():
    """Predict one uploaded image and record latency metrics."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided."}), 400

    file_obj = request.files["file"]
    if file_obj.filename == "":
        return jsonify({"error": "No file selected."}), 400

    started = time.perf_counter()
    try:
        result = predict_from_uploaded_file(file_obj)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

    latency_ms = (time.perf_counter() - started) * 1000.0
    METRICS["total_predictions"] += 1
    METRICS["total_latency_ms"] += latency_ms
    METRICS["last_prediction_at"] = datetime.utcnow().isoformat()

    return jsonify(
        {
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "raw_probability": result["raw_probability"],
            "latency_ms": round(latency_ms, 2),
        }
    )


@app.route("/upload-retrain-data", methods=["POST"])
def upload_retrain_data():
    """Store bulk user uploads for later retraining."""
    class_label = request.form.get("class_label", "").strip().upper()
    files = request.files.getlist("files")

    if class_label not in CLASS_NAMES:
        return jsonify({"error": f"class_label must be one of: {CLASS_NAMES}"}), 400
    if not files:
        return jsonify({"error": "No files were uploaded."}), 400

    try:
        saved = save_uploaded_files(files, class_label=class_label, uploads_dir=UPLOADS_DIR)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify(
        {
            "message": "Files uploaded successfully.",
            "class_label": class_label,
            "saved_count": len(saved),
        }
    )


@app.route("/trigger-retrain", methods=["POST"])
def trigger_retrain():
    """Start background retraining with provided epochs."""
    payload = request.get_json(silent=True) or {}
    requested_epochs = int(payload.get("epochs", 3))
    max_ui_epochs = max(1, int(os.getenv("UI_RETRAIN_MAX_EPOCHS", "2")))
    epochs = max(1, min(requested_epochs, max_ui_epochs))

    with STATUS_LOCK:
        if RETRAIN_STATUS["state"] == "running":
            return jsonify({"error": "Retraining is already running."}), 409

    worker = threading.Thread(target=_retrain_worker, args=(epochs,), daemon=True)
    worker.start()
    return jsonify({"message": "Retraining triggered.", "epochs": epochs})


@app.route("/retrain-status", methods=["GET"])
def retrain_status():
    """Return the latest retraining state and result payload."""
    with STATUS_LOCK:
        return jsonify(RETRAIN_STATUS)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)