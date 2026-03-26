import json
import os
import sys


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from preprocessing import CLASS_NAMES, dataset_split_status  # noqa: E402


def main():
    data_dir = os.path.join(ROOT_DIR, "data")
    status = dataset_split_status(data_dir)

    print(json.dumps(status, indent=2))

    errors = []
    for split in ["train", "val", "test"]:
        split_info = status.get(split, {})
        if not split_info.get("exists"):
            errors.append(f"Missing split directory: {split}")
            continue

        counts = split_info.get("class_counts", {})
        for class_name in CLASS_NAMES:
            if counts.get(class_name, 0) <= 0:
                errors.append(f"{split}/{class_name} has 0 images")

    if errors:
        print("\nDataset validation failed:")
        for err in errors:
            print(f"- {err}")
        return 1

    print("\nDataset validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
