from flask import Flask, request, jsonify, send_from_directory
from pathlib import Path
import base64, numpy as np, cv2
from keras.models import load_model

app = Flask(__name__, static_folder="web", static_url_path="")

MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "bakery_cnn.h5"
LABELS_PATH = MODELS_DIR / "labels.txt"

print("Loading model...")
try:
    model = load_model(MODEL_PATH.as_posix())
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f if line.strip()]
except Exception as e:
    print("Error loading model:", e)
    model, labels = None, None

def decode_b64_image(image_data: str):
    if not image_data: return None
    if ',' in image_data: image_data = image_data.split(',', 1)[1]
    try:
        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception:
        return None

def center_crop_square(img_rgb):
    h, w = img_rgb.shape[:2]; s = min(h, w)
    y0 = (h - s)//2; x0 = (w - s)//2
    return img_rgb[y0:y0+s, x0:x0+s]

def preprocess_for_model(frame_bgr, target_size=(224,224), center_crop=True):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    if center_crop: rgb = center_crop_square(rgb)
    img = cv2.resize(rgb, target_size).astype(np.float32)/255.0
    return np.expand_dims(img, 0)

def predict_frame(frame_bgr, topk=3, threshold=0.60, center_crop=True):
    x = preprocess_for_model(frame_bgr, center_crop=center_crop)
    probs = model.predict(x, verbose=0)[0]
    idx = np.argsort(probs)[::-1][:min(topk, len(probs))]
    pairs = [(labels[i], float(probs[i])) for i in idx]
    best_label, best_conf = pairs[0]
    text = f"{best_label} ({best_conf:.2f})" if best_conf >= threshold else \
           "Không chắc chắn (%.2f). Gợi ý: %s" % (best_conf, ", ".join(f"{n}({c:.2f})" for n,c in pairs))
    return {"prediction": text, "label": best_label, "confidence": best_conf, "topk": pairs}

@app.after_request
def add_cors(resp):
    resp.headers.add('Access-Control-Allow-Origin', '*')
    resp.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    resp.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return resp

@app.route("/")
def index(): return send_from_directory(app.static_folder, "index.html")

@app.route("/<path:p>")
def static_proxy(p): return send_from_directory(app.static_folder, p)

@app.route("/predict", methods=["POST","OPTIONS"])
def predict():
    if request.method == "OPTIONS": return jsonify({})
    data = request.get_json(force=True, silent=True) or {}
    frame = decode_b64_image(data.get("image"))
    if frame is None: return jsonify({"prediction":"Invalid image", "error":True}), 400
    center_crop = bool(data.get("center_crop", True))
    out = predict_frame(frame, center_crop=center_crop)
    return jsonify({"prediction": out["prediction"], "label": out["label"], "confidence": out["confidence"], "topk": out["topk"], "error": False})

@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    data = request.get_json(force=True, silent=True) or {}
    images = data.get("images", []); center_crop = bool(data.get("center_crop", True))
    results = []
    for b64 in images:
        frame = decode_b64_image(b64)
        if frame is None: results.append({"prediction":"Invalid image", "error":True}); continue
        out = predict_frame(frame, center_crop=center_crop)
        results.append({"prediction": out["prediction"], "label": out["label"], "confidence": out["confidence"], "topk": out["topk"], "error": False})
    return jsonify({"results": results, "error": False})

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
