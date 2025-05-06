import os
import time
import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, request, jsonify, render_template

# تحميل النموذج وملف labels
model = tf.keras.models.load_model('keras_model.h5')
with open('labels.txt', 'r', encoding='utf-8') as f:
    class_names = [line.strip() for line in f.readlines()]

app = Flask(__name__)

current_label = ""
label_start_time = 0
spoken = False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video', methods=['POST'])
def video():
    global current_label, label_start_time, spoken

    file = request.files.get('frame')
    if not file:
        print("❌ لم يتم استلام أي ملف صورة.")
        return 'No frame received', 400

    try:
        # قراءة الصورة
        npimg = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if img is None:
            print("❌ فشل في تحويل الصورة (img is None).")
            return 'Failed to decode image', 400

        # تجهيز الصورة للنموذج
        img = cv2.resize(img, (224, 224))
        img = np.expand_dims(img, axis=0)
        img = img.astype(np.float32)
        img = (img / 127.5) - 1

        # التنبؤ
        prediction = model.predict(img)
        index = np.argmax(prediction)
        label = class_names[index]

        print(f"✅ التنبؤ: {label} - الدقة: {np.max(prediction):.2f}")

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
        print(f"❌ خطأ في المعالجة: {str(e)}")
        return 'Internal Server Error', 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
