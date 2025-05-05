from flask import Flask, request, send_file
import cv2
import numpy as np
import mediapipe as mp
import time
import os
import tensorflow as tf

app = Flask(__name__)

# تحميل النموذج
model = tf.keras.models.load_model('keras_model.h5')

# قراءة التسميات من الملف labels.txt
with open("labels.txt", "r", encoding="utf-8") as f:
    class_names = f.read().splitlines()

# إعداد mediapipe لاكتشاف اليد
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

# متغيرات لمتابعة الحرف الذي تم التعرّف عليه
current_label = ""
label_start_time = 0
spoken = False

@app.route('/')
def index():
    return send_file('index2.html')  # تأكد أن الاسم مطابق للواجهة لديك

@app.route('/video', methods=['POST'])
def video():
    global current_label, label_start_time, spoken
    try:
        file = request.files['frame']
        if not file:
            return '', 400

        # معالجة الصورة المرسلة
        npimg = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            h, w, _ = img.shape
            hand_points = []
            for lm in hand_landmarks.landmark:
                hand_points.extend([lm.x, lm.y, lm.z])

            input_data = np.array([hand_points], dtype=np.float32)
            prediction = model.predict(input_data)
            predicted_index = np.argmax(prediction)
            label = class_names[predicted_index] if predicted_index < len(class_names) else "?"

            # تأكيد ثبات الحرف قبل النطق
            if label != current_label:
                current_label = label
                label_start_time = time.time()
                spoken = False
            else:
                elapsed = time.time() - label_start_time
                if elapsed >= 2 and not spoken:
                    spoken = True
                    return label, 200

        return '', 204  # لا يوجد حرف جديد للتحدث
    except Exception as e:
        print("Error:", e)
        return 'Internal Server Error', 500

# تشغيل السيرفر سواء على Render أو محلياً
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
