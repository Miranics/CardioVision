"""Model training and retraining utilities for CardioVision."""

import os
from datetime import datetime

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
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
	"""Build a MobileNetV2 transfer-learning classifier."""
	base_model = MobileNetV2(
		include_top=False,
		weights="imagenet",
		input_shape=input_shape,
	)
	base_model.trainable = False

	x = GlobalAveragePooling2D()(base_model.output)
	x = Dropout(0.3)(x)
	output = Dense(1, activation="sigmoid")(x)

	model = Model(inputs=base_model.input, outputs=output)
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


def train_model(data_dir, model_output_path, epochs=5, learning_rate=1e-4, batch_size=32):
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
	callbacks = [
		EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
		ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
	]

	history = model.fit(
		train_gen,
		validation_data=val_gen,
		epochs=epochs,
		callbacks=callbacks,
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


def retrain_from_uploaded_data(
	base_data_dir,
	uploads_dir,
	models_dir,
	epochs=3,
	learning_rate=1e-4,
	batch_size=8,
):
	"""Merge uploaded files into train set, retrain model, and return run summary."""
	train_dir = os.path.join(base_data_dir, "train")
	copied_files = merge_uploads_into_training_data(uploads_dir, train_dir)
	if copied_files == 0:
		raise ValueError("No uploaded files found to retrain with.")

	timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
	output_path = os.path.join(models_dir, f"cardiovision_model_retrained_{timestamp}.keras")

	training_result = train_model(
		data_dir=base_data_dir,
		model_output_path=output_path,
		epochs=epochs,
		learning_rate=learning_rate,
		batch_size=batch_size,
	)
	training_result["copied_training_files"] = copied_files
	return training_result
