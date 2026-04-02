"""Model training and retraining utilities for CardioVision."""

import os
from datetime import datetime

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from preprocessing import (
	CLASS_NAMES,
	IMG_SIZE,
	build_data_generators,
	dataset_split_status,
	merge_uploads_into_training_data,
)


def _validate_dataset_for_training(data_dir):
	"""Ensure required splits and classes exist before fitting a model."""
	status = dataset_split_status(data_dir)
	missing = []

	for split in ["train", "val", "test"]:
		split_info = status.get(split, {})
		if not split_info.get("exists"):
			missing.append(f"Missing split directory: {split_info.get('path', split)}")
			continue

		counts = split_info.get("class_counts", {})
		for class_name in CLASS_NAMES:
			if counts.get(class_name, 0) <= 0:
				missing.append(
					f"{split}/{class_name} has no images. Add at least one image in this class."
				)

	if missing:
		raise ValueError("Dataset is not ready for training: " + " | ".join(missing))


def build_transfer_model(input_shape=(224, 224, 3), learning_rate=1e-4):
	"""Build a lightweight CNN classifier for low-memory runtime retraining."""
	tf.keras.backend.clear_session()
	model = Sequential(
		[
			Conv2D(16, (3, 3), activation="relu", input_shape=input_shape),
			MaxPooling2D((2, 2)),
			Conv2D(32, (3, 3), activation="relu"),
			MaxPooling2D((2, 2)),
			Conv2D(64, (3, 3), activation="relu"),
			MaxPooling2D((2, 2)),
			Flatten(),
			Dense(64, activation="relu"),
			Dropout(0.3),
			Dense(1, activation="sigmoid"),
		]
	)
	model.compile(
		optimizer=Adam(learning_rate=learning_rate),
		loss="binary_crossentropy",
		metrics=["accuracy"],
	)
	return model


def evaluate_binary_model(model, test_generator):
	"""Evaluate trained model using accuracy, precision, recall, and F1."""
	predictions = model.predict(test_generator, verbose=0)
	y_probs = predictions.reshape(-1)
	y_pred = (y_probs >= 0.5).astype(int)
	y_true = test_generator.classes

	return {
		"accuracy": float(accuracy_score(y_true, y_pred)),
		"precision": float(precision_score(y_true, y_pred, zero_division=0)),
		"recall": float(recall_score(y_true, y_pred, zero_division=0)),
		"f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
	}


def train_model(
	data_dir,
	model_output_path,
	epochs=5,
	learning_rate=1e-4,
	batch_size=32,
	callbacks=None,
):
	"""Train, evaluate, and save a classification model."""
	_validate_dataset_for_training(data_dir)
	train_gen, val_gen, test_gen = build_data_generators(
		data_dir,
		img_size=IMG_SIZE,
		batch_size=batch_size,
	)

	model = build_transfer_model(
		input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
		learning_rate=learning_rate,
	)
	default_callbacks = [
		EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
		ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
	]
	all_callbacks = default_callbacks + list(callbacks or [])

	history = model.fit(
		train_gen,
		validation_data=val_gen,
		epochs=epochs,
		callbacks=all_callbacks,
		verbose=1,
	)

	os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
	model.save(model_output_path)

	metrics = evaluate_binary_model(model, test_gen)
	return {
		"saved_model_path": model_output_path,
		"epochs_ran": len(history.history.get("loss", [])),
		"metrics": metrics,
	}


class _TrainingProgressCallback(Callback):
	"""Push epoch-level status messages back to caller."""

	def __init__(self, total_epochs, progress_callback):
		super().__init__()
		self.total_epochs = max(1, int(total_epochs))
		self.progress_callback = progress_callback

	def on_epoch_end(self, epoch, logs=None):
		if not self.progress_callback:
			return
		loss = None if logs is None else logs.get("loss")
		val_loss = None if logs is None else logs.get("val_loss")
		loss_txt = "?" if loss is None else f"{float(loss):.4f}"
		val_loss_txt = "?" if val_loss is None else f"{float(val_loss):.4f}"
		self.progress_callback(
			f"Epoch {epoch + 1}/{self.total_epochs} completed (loss={loss_txt}, val_loss={val_loss_txt})."
		)


def retrain_from_uploaded_data(
	base_data_dir,
	uploads_dir,
	models_dir,
	epochs=3,
	learning_rate=1e-4,
	batch_size=8,
	progress_callback=None,
):
	"""Merge uploaded files into train set, retrain model, and return run summary."""
	if progress_callback:
		progress_callback("Preparing uploaded files for retraining.")
	train_dir = os.path.join(base_data_dir, "train")
	copied_files = merge_uploads_into_training_data(uploads_dir, train_dir)
	if copied_files == 0:
		raise ValueError("No uploaded files found to retrain with.")
	if progress_callback:
		progress_callback(f"Merged {copied_files} uploaded files into the train split.")

	status = dataset_split_status(base_data_dir)
	train_counts = status.get("train", {}).get("class_counts", {})
	if progress_callback:
		progress_callback(
			"Training uses the full current train split: "
			f"NORMAL={train_counts.get('NORMAL', 0)}, "
			f"PNEUMONIA={train_counts.get('PNEUMONIA', 0)}."
		)

	timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
	output_path = os.path.join(models_dir, f"cardiovision_model_retrained_{timestamp}.keras")

	if progress_callback:
		progress_callback("Starting model training. This can be memory-intensive on small instances.")

	callbacks = None
	if progress_callback:
		callbacks = [_TrainingProgressCallback(total_epochs=epochs, progress_callback=progress_callback)]

	training_result = train_model(
		data_dir=base_data_dir,
		model_output_path=output_path,
		epochs=epochs,
		learning_rate=learning_rate,
		batch_size=batch_size,
		callbacks=callbacks,
	)
	if progress_callback:
		progress_callback(f"Training completed. Saved model to {output_path}.")
	training_result["copied_training_files"] = copied_files
	return training_result
