
import os
import io
import json
import base64
import sys
import numpy as np
import cv2
from flask import Flask, request, jsonify
from ultralytics import YOLO
import easyocr
from deepface import DeepFace
from PIL import Image

app = Flask(__name__)

# -------------------- Model Initialization --------------------
# We load models at startup so they remain in memory (Free Spaces have 16GB RAM, ample space)
print("Loading KYC Models...", file=sys.stderr)

# 1. OCR
reader = easyocr.Reader(['en', 'ur'], gpu=False)

# 2. YOLO
yolo_model = YOLO('yolov8n.pt')

# 3. DeepFace: Lazy loaded usually, but we can verify it loads by a dummy call or just wait.
# It downloads weights to ~/.deepface/weights on first run.

print("Models loaded!", file=sys.stderr)

# -------------------- Helper Functions --------------------

def decode_image(image_input):
    """Decodes image from base64 string or file bytes."""
    try:
        # If it's a base64 string
        if isinstance(image_input, str) and "," in image_input:
            image_input = image_input.split(",")[1]
        
        if isinstance(image_input, str):
            img_bytes = base64.b64decode(image_input)
        else:
            img_bytes = image_input # Assume bytes

        img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        raise ValueError(f"Invalid image input: {e}")

def crop_card(image):
    results = yolo_model(image, verbose=False)
    best_box = None
    max_area = 0
    
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            area = (x2 - x1) * (y2 - y1)
            if area > max_area:
                max_area = area
                best_box = (int(x1), int(y1), int(x2), int(y2))
    
    if best_box:
        x1, y1, x2, y2 = best_box
        h, w, _ = image.shape
        x1 = max(0, x1 - 10)
        y1 = max(0, y1 - 10)
        x2 = min(w, x2 + 10)
        y2 = min(h, y2 + 10)
        return image[y1:y2, x1:x2]
    
    return image

# -------------------- Endpoints --------------------

@app.route('/', methods=['GET'])
def health():
    return "KYC Engine is Running"

@app.route('/verify-cnic', methods=['POST'])
def verify_cnic():
    try:
        data = request.json
        image_data = data.get('image')
        if not image_data:
            return jsonify({"error": "No image provided"}), 400

        img = decode_image(image_data)
        cropped = crop_card(img)
        
        # OCR
        text_results = reader.readtext(cropped, detail=0)
        
        return jsonify({
            "raw_text": text_results
            # We let Node.js do the regex parsing to keep Python lighter if desired, 
            # or we can move the extract_cnic_info logic here. 
            # For now, let's return raw text so Node can decide how to use it.
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/face-verify', methods=['POST'])
def face_verify():
    try:
        data = request.json
        img1_data = data.get('image1') # CNIC
        img2_data = data.get('image2') # Selfie
        
        if not img1_data or not img2_data:
            return jsonify({"error": "image1 and image2 required"}), 400

        # DeepFace expects paths or numpy arrays (BGR)
        img1 = decode_image(img1_data)
        img2 = decode_image(img2_data)

        result = DeepFace.verify(
            img1_path=img1, 
            img2_path=img2, 
            model_name='VGG-Face', 
            enforce_detection=False
        )
        return jsonify({
            "verified": result['verified'],
            "distance": result['distance'],
            "threshold": result['threshold']
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/shop-verify', methods=['POST'])
def shop_verify():
    try:
        data = request.json
        image_data = data.get('image')
        if not image_data:
            return jsonify({"error": "No image provided"}), 400
        
        img = decode_image(image_data)
        
        # Detect objects
        results = yolo_model(img, verbose=False)
        detected_objects = []
        for r in results:
            for box in r.boxes:
                 cls_id = int(box.cls[0])
                 cls_name = yolo_model.names[cls_id]
                 conf = float(box.conf[0])
                 if conf > 0.4:
                     detected_objects.append(cls_name)
        
        # OCR on shop doc/signboard
        ocr_text = reader.readtext(img, detail=0)
        
        return jsonify({
            "detected_objects": list(set(detected_objects)),
            "text_content": ocr_text
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Run on port 7860 (Hugging Face default)
    app.run(host='0.0.0.0', port=7860)
