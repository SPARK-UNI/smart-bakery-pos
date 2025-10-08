# app.py
from flask import Flask, request, jsonify, send_from_directory
from pathlib import Path
import base64
import numpy as np
import cv2
from keras.models import load_model

app = Flask(__name__, static_folder="web", static_url_path="")

# ====== Đường dẫn model & labels ======
MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "bakery_cnn.h5"
LABELS_PATH = MODELS_DIR / "labels.txt"

# ====== Nạp model & labels ======
print("Loading model...")
try:
    model = load_model(MODEL_PATH.as_posix())
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f if line.strip()]
    print(f"Model loaded: {MODEL_PATH}")
    print(f"Labels ({len(labels)}): {labels}")
except Exception as e:
    print("Error loading model/labels:", e)
    model, labels = None, None

# ====== Tiện ích ======
def decode_b64_image(image_data: str):
    """Nhận base64 dataURL hoặc base64 thuần -> OpenCV BGR image."""
    if not image_data:
        return None
    if ',' in image_data:  # kiểu data:image/jpeg;base64,....
        image_data = image_data.split(',', 1)[1]
    try:
        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # BGR
        return frame
    except Exception:
        return None

def center_crop_square(img_rgb: np.ndarray):
    """Crop vuông ở giữa để giảm nền, sau đó trả ảnh RGB đã crop."""
    h, w = img_rgb.shape[:2]
    s = min(h, w)
    y0 = (h - s) // 2
    x0 = (w - s) // 2
    return img_rgb[y0:y0+s, x0:x0+s]

def preprocess_for_model(frame_bgr: np.ndarray, target_size=(224, 224)):
    """BGR -> RGB -> center-crop -> resize -> scale [0,1] -> (1,H,W,3)."""
    # 1) BGR -> RGB (QUAN TRỌNG vì Keras/PIL dùng RGB khi train)
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    # 2) Center crop vuông (giảm nền/khung cảnh)
    crop = center_crop_square(rgb)
    # 3) Resize & scale
    img = cv2.resize(crop, target_size).astype(np.float32) / 255.0
    # 4) Add batch dimension
    return np.expand_dims(img, axis=0)

def predict_frame(frame_bgr: np.ndarray, topk: int = 3, threshold: float = 0.60) -> str:
    """Dự đoán 1 frame. Trả về text thân thiện cho UI."""
    try:
        x = preprocess_for_model(frame_bgr, target_size=(224, 224))
        probs = model.predict(x, verbose=0)[0]  # shape: (num_classes,)
        # Top-k
        idx_sorted = np.argsort(probs)[::-1]
        idx_topk = idx_sorted[:min(topk, len(probs))]
        pairs = [(labels[i], float(probs[i])) for i in idx_topk]
        best_label, best_conf = pairs[0]

        if best_conf < threshold:
            # Không chắc chắn: trả gợi ý top-k
            hints = ", ".join([f"{n}({c:.2f})" for n, c in pairs])
            return f"Không chắc chắn ({best_conf:.2f}). Gợi ý: {hints}"
        else:
            return f"{best_label} ({best_conf:.2f})"
    except Exception as e:
        return f"Error: {str(e)}"

# ====== CORS đơn giản để dev ======
@app.after_request
def add_cors_headers(resp):
    resp.headers.add('Access-Control-Allow-Origin', '*')
    resp.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    resp.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return resp

# ====== Serve UI ======
@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/<path:path>")
def static_proxy(path):
    return send_from_directory(app.static_folder, path)

# ====== API predict ======
@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return jsonify({})

    if model is None or labels is None:
        return jsonify({"prediction": "Model not loaded", "error": True})

    data = request.get_json(force=True, silent=True) or {}
    image_data = data.get("image")
    frame = decode_b64_image(image_data)
    if frame is None:
        return jsonify({"prediction": "Invalid image", "error": True})

    result = predict_frame(frame, topk=3, threshold=0.60)
    return jsonify({"prediction": result, "error": result.startswith("Error:")})

if __name__ == "__main__":
    print("Server running at: http://127.0.0.1:5000")
    app.run(debug=True, host="127.0.0.1", port=5000)
