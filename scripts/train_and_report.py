import json
import os
import sys
from datetime import datetime


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from model import train_model  # noqa: E402


def main():
    epochs = int(os.getenv("CV_TRAIN_EPOCHS", "3"))
    learning_rate = float(os.getenv("CV_TRAIN_LR", "1e-4"))

    data_dir = os.path.join(ROOT_DIR, "data")
    models_dir = os.path.join(ROOT_DIR, "models")
    reports_dir = os.path.join(ROOT_DIR, "reports")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    model_output_path = os.path.join(models_dir, f"cardiovision_model_run_{timestamp}.keras")

    result = train_model(
        data_dir=data_dir,
        model_output_path=model_output_path,
        epochs=epochs,
        learning_rate=learning_rate,
    )

    payload = {
        "timestamp_utc": datetime.utcnow().isoformat(),
        "epochs": epochs,
        "learning_rate": learning_rate,
        "result": result,
    }

    report_path = os.path.join(reports_dir, f"training_report_{timestamp}.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(json.dumps(payload, indent=2))
    print(f"Saved report to: {report_path}")


if __name__ == "__main__":
    main()
