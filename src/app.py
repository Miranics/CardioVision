from flask import Flask, request, jsonify
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Load model
MODEL_PATH = "../models/cardiovision_model_retrained.keras"
model = load_model(MODEL_PATH)

# Class labels
CLASS_NAMES = ['NORMAL', 'PNEUMONIA']


def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


@app.route('/')
def home():
    return "CardioVision API is running"


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    filepath = os.path.join("temp", file.filename)
    os.makedirs("temp", exist_ok=True)
    file.save(filepath)

    img = preprocess_image(filepath)
    prediction = model.predict(img)[0][0]

    label = CLASS_NAMES[1] if prediction > 0.5 else CLASS_NAMES[0]
    confidence = float(prediction if prediction > 0.5 else 1 - prediction)

    os.remove(filepath)

    return jsonify({
        'prediction': label,
        'confidence': round(confidence, 4)
    })


if __name__ == '__main__':
    app.run(debug=True)