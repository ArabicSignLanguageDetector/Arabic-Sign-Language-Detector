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

# تحميل أسماء الحروف
with open("labels.txt", "r", encoding="utf-8") as f:
    class_names = [line.split(' ', 1)[1] for line in f if ' ' in line]

# إعداد mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# تتبع النطق
current_label = ""
label_start_time = 0
spoken = False

@app.route('/')
def index():
    return send_file('index.html')

@app.route('/video', methods=['POST'])
def video():
    global current_label, label_start_time, spoken

    print("📷 Frame received")  # خطوة 1: تأكيد وصول الصورة

    try:
        file = request.files['frame']
        if not file:
            print("❌ No frame found")
            return '', 400

        npimg = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if img is None:
            print("❌ Failed to decode image")  # خطوة 2: التأكد من قراءة الصورة
            return 'Failed to decode image', 500

        # اختياري: حفظ الصورة لفحصها يدويًا (احذفها لاحقًا)
        cv2.imwrite("debug_frame.jpg", img)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            hand_points = [val for lm in hand_landmarks.landmark for val in (lm.x, lm.y, lm.z)]

            input_data = np.array([hand_points], dtype=np.float32)
            prediction = model.predict(input_data)
            predicted_index = np.argmax(prediction)
            label = class_names[predicted_index] if predicted_index < len(class_names) else "?"

            print(f"🧠 Prediction: {prediction}, Label: {label}")  # خطوة 3: التحقق من ناتج النموذج

            if label != current_label:
                current_label = label
                label_start_time = time.time()
                spoken = False
            else:
                elapsed = time.time() - label_start_time
                if elapsed >= 2 and not spoken:
                    spoken = True
                    if label != "?":
                        print(f"🔊 Sending label to client: {label}")
                        return label, 200

        else:
            print("🖐️ No hand detected")

        return '', 204

    except Exception as e:
        print(f"❌ Internal server error: {e}")
        return 'Internal Server Error', 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
