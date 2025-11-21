from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import joblib
from collections import deque
import threading
import os

app = Flask(__name__)

# ---------------- Load Model and Encoder ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = tf.keras.models.load_model(r"D:\Sign-Language-Detection\project\Indian-Sign-Language\scripts\dataset\sign_language_mobilenet.h5")
le = joblib.load(r"D:\Sign-Language-Detection\project\Indian-Sign-Language\scripts\dataset\label_encoder_mobilenet.pkl")


# ---------------- MediaPipe Setup ----------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ---------------- Globals ----------------
cap = None
is_running = False
sentence = ""
prev_char = ""
buffer = deque(maxlen=15)
current_letter = ""   # ✅ NEW: to store live letter


# ---------------- Frame Generator ----------------
def generate_frames():
    global cap, is_running, sentence, prev_char, buffer, current_letter

    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(max_num_hands=1,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5) as hands:
        while is_running:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # Process frame with MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Get bounding box of hand
                    x_coords = [int(lm.x * w) for lm in hand_landmarks.landmark]
                    y_coords = [int(lm.y * h) for lm in hand_landmarks.landmark]
                    x_min, x_max = max(min(x_coords) - 20, 0), min(max(x_coords) + 20, w)
                    y_min, y_max = max(min(y_coords) - 20, 0), min(max(y_coords) + 20, h)

                    hand_img = frame[y_min:y_max, x_min:x_max]
                    if hand_img.size == 0:
                        continue

                    # Preprocess ROI (same as training)
                    roi = cv2.resize(hand_img, (128, 128))
                    roi = roi.astype("float32") / 255.0
                    roi = np.expand_dims(roi, axis=0)

                    # Predict
                    preds = model.predict(roi, verbose=0)
                    class_id = np.argmax(preds)
                    confidence = preds[0][class_id]
                    label = le.inverse_transform([class_id])[0]

                    # Only consider confident predictions
                    if confidence > 0.7:
                        buffer.append(label)

                    # Stabilize prediction
                    if len(buffer) == buffer.maxlen:
                        most_common = max(set(buffer), key=buffer.count)
                        current_letter = most_common  # ✅ NEW: update live letter
                        if most_common != prev_char:
                            if most_common == "SPACE":
                                sentence += " "
                            elif most_common == "DEL":
                                sentence = sentence[:-1]
                            else:
                                sentence += most_common
                            prev_char = most_common

                    # Draw annotations
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} ({confidence*100:.1f}%)",
                                (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.9, (0, 0, 255), 2)

            # Display sentence
            # cv2.putText(frame, f"Sentence: {sentence}", (30, 450),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Encode for streaming
            ret, buffer_frame = cv2.imencode('.jpg', frame)
            frame = buffer_frame.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


# ---------------- Flask Routes ----------------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start', methods=['POST'])
def start():
    global is_running
    if not is_running:
        is_running = True
        threading.Thread(target=generate_frames, daemon=True).start()
    return jsonify({"message": "Detection started"})


@app.route('/stop', methods=['POST'])
def stop():
    global is_running, sentence, prev_char, buffer, current_letter
    is_running = False
    sentence = ""
    prev_char = ""
    current_letter = ""
    buffer.clear()
    return jsonify({"message": "Detection stopped"})


# ✅ NEW: Clear sentence route
@app.route('/clear', methods=['POST'])
def clear():
    global sentence, current_letter
    sentence = ""
    current_letter = ""
    return jsonify({"message": "Sentence cleared"})


# ✅ UPDATED: Return current letter + sentence
@app.route('/get_sentence')
def get_sentence():
    global sentence, current_letter
    return jsonify({"sentence": sentence, "current": current_letter})


# ---------------- Run App ----------------
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000,debug=True)
