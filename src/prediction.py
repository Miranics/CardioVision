"""Inference helpers for loading model and predicting from uploaded images."""

import os
import threading
from io import BytesIO

import numpy as np
from PIL import Image, UnidentifiedImageError


IMG_SIZE = (224, 224)
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(6 * 1024 * 1024)))
MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", "25000000"))

Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

_model = None
_model_path = None
_model_lock = threading.Lock()


def set_model_path(model_path):
	"""Set active model path and clear cached model instance."""
	global _model, _model_path
	_model = None
	_model_path = model_path


def get_model(model_path=None):
	"""Return loaded model, reloading only when path changes."""
	global _model, _model_path

	if model_path and model_path != _model_path:
		_model = None
		_model_path = model_path

	if _model is None:
		with _model_lock:
			if _model is None:
				if not _model_path:
					raise ValueError("Model path is not configured.")
				from tensorflow.keras.models import load_model

				# Inference-only load avoids allocating training/optimizer state.
				_model = load_model(_model_path, compile=False)
	return _model


def preprocess_uploaded_image(file_storage, target_size=IMG_SIZE):
	"""Decode and normalize one uploaded image for model inference."""
	file_bytes = file_storage.read()
	if not file_bytes:
		raise ValueError("Uploaded file is empty.")
	if len(file_bytes) > MAX_IMAGE_BYTES:
		raise ValueError(
			f"Uploaded file is too large. Maximum size is {MAX_IMAGE_BYTES // (1024 * 1024)} MB."
		)

	try:
		pil_img = Image.open(BytesIO(file_bytes)).convert("RGB")
	except UnidentifiedImageError as exc:
		raise ValueError("Uploaded file is not a valid image.") from exc

	pil_img = pil_img.resize(target_size)
	img_array = np.array(pil_img, dtype=np.float32)
	img_array = np.expand_dims(img_array, axis=0)
	return img_array / 255.0


def predict_from_uploaded_file(file_storage, model_path=None):
	"""Predict class and confidence for one uploaded image file."""
	model = get_model(model_path)

	target_h = IMG_SIZE[0]
	target_w = IMG_SIZE[1]
	if hasattr(model, "input_shape") and model.input_shape is not None:
		if len(model.input_shape) >= 3:
			model_h = model.input_shape[1]
			model_w = model.input_shape[2]
			if isinstance(model_h, int) and isinstance(model_w, int):
				target_h = model_h
				target_w = model_w

	img_tensor = preprocess_uploaded_image(file_storage, target_size=(target_h, target_w))

	pred_prob = float(model(img_tensor, training=False).numpy()[0][0])
	pred_class = CLASS_NAMES[1] if pred_prob >= 0.5 else CLASS_NAMES[0]
	confidence = pred_prob if pred_class == CLASS_NAMES[1] else 1.0 - pred_prob

	return {
		"prediction": pred_class,
		"confidence": round(confidence, 4),
		"raw_probability": round(pred_prob, 4),
	}
