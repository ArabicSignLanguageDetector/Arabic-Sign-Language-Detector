from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import math
import os
import base64
from io import BytesIO
from PIL import Image
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import time

app = Flask(__name__)

model_path = "keras_model.h5"
labels_path = "labels.txt"

# Load labels
labels = []
with open(labels_path, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            parts = line.strip().split(' ', 1)
            labels.append(parts[1] if len(parts) == 2 else parts[0])

# Init detector and classifier
detector = HandDetector(maxHands=1)
classifier = Classifier(model_path, labels_path)

offset = 20
imgSize = 300

@app.route('/')
def index():
    return render_template("index.html")

# ✅ route الجديد المتوافق مع الكاميرا في HTML
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400

    try:
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        h_img, w_img, _ = img.shape
        x1 = max(0, x - offset)
        y1 = max(0, y - offset)
        x2 = min(x + w + offset, w_img)
        y2 = min(y + h + offset, h_img)
        imgCrop = img[y1:y2, x1:x2]

        if imgCrop.size == 0:
            return '', 204

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wGap + wCal] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hGap + hCal, :] = imgResize

        prediction, index = classifier.getPrediction(imgWhite, draw=False)
        if index < len(labels):
            label = labels[index]
            confidence = float(prediction[index])
            return jsonify({
                'label': label,
                'confidence': confidence
            }), 200

    return '', 204

# ✅ route القديم (اختياري – للإبقاء على التوافق مع نسخ سابقة)
@app.route('/video', methods=['POST'])
def video():
    file = request.files.get('frame')
    if not file:
        return jsonify({'error': 'No frame received'}), 400

    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        h_img, w_img, _ = img.shape
        x1 = max(0, x - offset)
        y1 = max(0, y - offset)
        x2 = min(x + w + offset, w_img)
        y2 = min(y + h + offset, h_img)

        imgCrop = img[y1:y2, x1:x2]
        if imgCrop.size == 0:
            return '', 204

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wGap + wCal] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hGap + hCal, :] = imgResize

        prediction, index = classifier.getPrediction(imgWhite, draw=False)
        if index < len(labels):
            label = labels[index]
            confidence = float(prediction[index])
            return jsonify({
                'label': label,
                'confidence': confidence
            }), 200

    return '', 204

# Run the app
port = int(os.environ.get("PORT", 10000))
app.run(host='0.0.0.0', port=port)
