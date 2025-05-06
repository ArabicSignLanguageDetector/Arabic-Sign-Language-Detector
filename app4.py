import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, send_file
import time
import os

# تحميل النموذج
model = tf.keras.models.load_model('keras_model.h5')

# قراءة التصنيفات
with open("labels.txt", "r", encoding="utf-8") as f:
    labels = [line.strip() for line in f.readlines()]

app = Flask(__name__)

# متغيرات الحالة
current_label = ""
label_start_time = 0
spoken = False

@app.route('/')
def index():
    return send_file('index.html')

@app.route('/video', methods=['POST'])
def video():
    global current_label, label_start_time, spoken

    file = request.files.get('frame')
    if not file:
        return '', 400

    # تحويل الملف إلى صورة OpenCV
    try:
        npimg = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if img is None:
            print("فشل في تحويل الصورة")
            return 'Failed to decode image', 500
    except Exception as e:
        print("خطأ أثناء قراءة الصورة:", e)
        return 'Error decoding image', 500

    try:
        # معالجة الصورة
        img = cv2.resize(img, (224, 224))
        img = np.asarray(img, dtype=np.float32).reshape(1, 224, 224, 3)
        img = (img / 127.5) - 1

        # التنبؤ
        prediction = model.predict(img, verbose=0)
        index = np.argmax(prediction)
        label = labels[index]

        # منطق التوقيت
        if label != current_label:
            current_label = label
            label_start_time = time.time()
            spoken = False
        else:
            elapsed = time.time() - label_start_time
            if elapsed >= 2 and not spoken:
                spoken = True
                return label, 200

        return '', 204
    except Exception as e:
        print("خطأ أثناء التنبؤ:", e)
        return 'Internal server error', 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
