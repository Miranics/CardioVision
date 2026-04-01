"""Dataset utilities for generation, status checks, and retraining uploads."""

import os
import shutil
from datetime import datetime
from pathlib import Path


IMG_SIZE = (224, 224)
BATCH_SIZE = 32
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def ensure_directory(path):
	"""Create a directory path if it does not exist."""
	Path(path).mkdir(parents=True, exist_ok=True)


def build_data_generators(data_dir, img_size=IMG_SIZE, batch_size=BATCH_SIZE):
	"""Create train/val/test image generators with augmentation for training."""
	from tensorflow.keras.preprocessing.image import ImageDataGenerator

	train_dir = os.path.join(data_dir, "train")
	val_dir = os.path.join(data_dir, "val")
	test_dir = os.path.join(data_dir, "test")

	for required in [train_dir, val_dir, test_dir]:
		if not os.path.isdir(required):
			raise FileNotFoundError(f"Missing required data directory: {required}")

	train_datagen = ImageDataGenerator(
		preprocessing_function=lambda x: x / 255.0,
		rotation_range=12,
		width_shift_range=0.1,
		height_shift_range=0.1,
		zoom_range=0.1,
		horizontal_flip=True,
	)
	eval_datagen = ImageDataGenerator(preprocessing_function=lambda x: x / 255.0)

	train_gen = train_datagen.flow_from_directory(
		train_dir,
		target_size=img_size,
		batch_size=batch_size,
		class_mode="binary",
		shuffle=True,
	)
	val_gen = eval_datagen.flow_from_directory(
		val_dir,
		target_size=img_size,
		batch_size=batch_size,
		class_mode="binary",
		shuffle=False,
	)
	test_gen = eval_datagen.flow_from_directory(
		test_dir,
		target_size=img_size,
		batch_size=batch_size,
		class_mode="binary",
		shuffle=False,
	)
	return train_gen, val_gen, test_gen


def count_images_by_class(dataset_split_dir):
	"""Count image files per class for a single dataset split."""
	split_path = Path(dataset_split_dir)
	counts = {class_name: 0 for class_name in CLASS_NAMES}
	if not split_path.exists():
		return counts

	for class_dir in split_path.iterdir():
		if class_dir.is_dir():
			total = 0
			for file_path in class_dir.iterdir():
				if file_path.suffix.lower() in ALLOWED_EXTENSIONS:
					total += 1
			counts[class_dir.name] = total
	return counts


def dataset_split_status(data_dir):
	"""Return split existence and class counts for train, val, and test."""
	status = {}
	for split in ["train", "val", "test"]:
		split_dir = Path(data_dir) / split
		status[split] = {
			"exists": split_dir.exists(),
			"path": str(split_dir),
			"class_counts": count_images_by_class(split_dir),
		}
	return status


def save_uploaded_files(files, class_label, uploads_dir):
	"""Save uploaded files into class-specific retraining intake folder."""
	normalized_label = class_label.strip().upper()
	if normalized_label not in CLASS_NAMES:
		raise ValueError(f"Invalid class label: {class_label}")

	target_dir = Path(uploads_dir) / normalized_label
	ensure_directory(target_dir)

	saved_paths = []
	for file_obj in files:
		if not file_obj.filename:
			continue

		extension = Path(file_obj.filename).suffix.lower()
		if extension not in ALLOWED_EXTENSIONS:
			continue

		timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
		safe_name = f"{timestamp}_{Path(file_obj.filename).name}"
		output_path = target_dir / safe_name
		file_obj.save(output_path)
		saved_paths.append(str(output_path))

	return saved_paths


def merge_uploads_into_training_data(uploads_dir, train_dir):
	"""Copy uploaded retrain files into train split with unique names."""
	uploads_path = Path(uploads_dir)
	train_path = Path(train_dir)
	ensure_directory(train_path)

	copied = 0
	for class_name in CLASS_NAMES:
		src_dir = uploads_path / class_name
		if not src_dir.exists():
			continue

		dest_dir = train_path / class_name
		ensure_directory(dest_dir)

		for file_path in src_dir.iterdir():
			if file_path.suffix.lower() not in ALLOWED_EXTENSIONS:
				continue

			unique_name = f"retrain_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}_{file_path.name}"
			shutil.copy2(file_path, dest_dir / unique_name)
			copied += 1

	return copied
