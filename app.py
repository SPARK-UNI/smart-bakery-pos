# app.py
from flask import Flask, request, jsonify, send_from_directory
import cv2
import numpy as np
from keras.models import load_model
import base64
from pathlib import Path

app = Flask(__name__)

MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "bakery_cnn.h5"
LABELS_PATH = MODELS_DIR / "labels.txt"

print("Loading model...")
try:
    model = load_model(MODEL_PATH.as_posix())
    print("Model loaded successfully!")

    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f if line.strip()]
    print(f"Labels loaded: {labels}")

except Exception as e:
    print(f"Error loading model or labels: {e}")
    model = None
    labels = None

def predict_from_image_b64(image_data):
    try:
        if ',' in image_data:
            image_data = image_data.split(',', 1)[1]
        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return "Invalid image"

        img = cv2.resize(frame, (224, 224)).astype(np.float32) / 255.0
        img = img.reshape(1, 224, 224, 3)

        pred = model.predict(img)
        class_idx = int(np.argmax(pred, axis=1)[0])
        confidence = float(np.max(pred, axis=1)[0])

        if 0 <= class_idx < len(labels):
            return f"{labels[class_idx]} ({confidence:.2f})"
        return "Unknown"
    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/')
def index():
    return send_from_directory('web', 'index.html')

@app.route('/style.css')
def css():
    return send_from_directory('web', 'style.css')

@app.route('/script.js')
def js():
    return send_from_directory('web', 'script.js')

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        response = jsonify({})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response

    try:
        if model is None or labels is None:
            response = jsonify({'prediction': 'Model not loaded', 'error': True})
        else:
            data = request.get_json(force=True, silent=True) or {}
            image_data = data.get('image')
            if not image_data:
                response = jsonify({'prediction': 'No image provided', 'error': True})
            else:
                prediction = predict_from_image_b64(image_data)
                response = jsonify({'prediction': prediction, 'error': False})

        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    except Exception as e:
        response = jsonify({'prediction': f'Server error: {str(e)}', 'error': True})
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response

if __name__ == '__main__':
    print("Server running at: http://127.0.0.1:5000")
    app.run(debug=True, host='127.0.0.1', port=5000)
