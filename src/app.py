import os
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "cardiovision_model_retrained.keras")
IMG_SIZE = (224, 224)
CLASS_NAMES = ['NORMAL', 'PNEUMONIA']

model = load_model(MODEL_PATH)
print("Model loaded successfully!")

app = Flask(__name__, template_folder="templates")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    img = image.load_img(file, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    prediction_prob = model.predict(img_array)[0][0]
    if prediction_prob >= 0.5:
        prediction_class = 'PNEUMONIA'
    else:
        prediction_class = 'NORMAL'

    confidence = float(prediction_prob) if prediction_class == 'PNEUMONIA' else 1 - float(prediction_prob)

    return jsonify({
        'prediction': prediction_class,
        'confidence': f"{confidence:.2f}"
    })

if __name__ == '__main__':
    app.run(debug=True)