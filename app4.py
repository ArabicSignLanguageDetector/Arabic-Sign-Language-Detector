from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import os
import time
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

app = Flask(__name__)

model_path = os.path.join("model", "keras_model.h5")
labels_path = os.path.join("model", "labels.txt")

labels = []
with open(labels_path, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            parts = line.strip().split(' ', 1)
            labels.append(parts[1] if len(parts) == 2 else parts[0])

detector = HandDetector(maxHands=1)
classifier = Classifier(model_path, labels_path)

offset = 20
imgSize = 300
current_label = ""
label_start_time = 0
spoken = False

@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/video', methods=['POST'])
def video():
    global current_label, label_start_time, spoken
    file = request.files['frame']
    if not file:
        return jsonify({'label': ''})

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
            return jsonify({'label': ''})

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = int(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = int((imgSize - wCal) / 2)
            imgWhite[:, wGap:wGap + wCal] = imgResize
        else:
            k = imgSize / w
            hCal = int(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = int((imgSize - hCal) / 2)
            imgWhite[hGap:hGap + hCal, :] = imgResize

        prediction, index = classifier.getPrediction(imgWhite, draw=False)
        label = labels[index] if index < len(labels) else "?"

        return jsonify({'label': label})

    return jsonify({'label': ''})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
