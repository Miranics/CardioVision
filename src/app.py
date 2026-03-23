import os
from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Absolute path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "cardiovision_model_retrained.keras")

# Load model
model = load_model(MODEL_PATH)

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
    
    temp_dir = os.path.join(BASE_DIR, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    filepath = os.path.join(temp_dir, file.filename)
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