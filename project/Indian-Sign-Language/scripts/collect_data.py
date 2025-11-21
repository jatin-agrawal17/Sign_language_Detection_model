import cv2
import mediapipe as mp
import os
from datetime import datetime

# ---------------- Dataset Path ----------------
dataset_folder = "dataset_images"
os.makedirs(dataset_folder, exist_ok=True)

# ---------------- Gestures to Collect ----------------
gestures = [chr(i) for i in range(65, 91)] + ["SPACE", "DEL"]  # A-Z + SPACE + DELETE

# Create folders if not exist
for g in gestures:
    os.makedirs(os.path.join(dataset_folder, g), exist_ok=True)

# ---------------- MediaPipe Setup ----------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

num_images = 400  # 📷 Recommended per gesture
print("Instructions:")
print("1️⃣ Show your gesture to the camera.")
print("2️⃣ Press the corresponding key (A-Z, SPACE=spacebar, DELETE=d).")
print("3️⃣ Press ESC to exit anytime.")

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.putText(frame, "Press A-Z | SPACE=spacebar | DEL=d | ESC=exit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Dataset Collector", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key != 255:
            if key == 32:
                label = "SPACE"
            elif key == ord('d'):
                label = "DEL"
            else:
                label = chr(key).upper()

            if label not in gestures:
                continue

            print(f"📸 Collecting for '{label}'...")
            saved = 0
            while saved < num_images:
                ret, frame = cap.read()
                if not ret:
                    continue
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    h, w, _ = frame.shape
                    x_coords = [int(lm.x * w) for lm in hand_landmarks.landmark]
                    y_coords = [int(lm.y * h) for lm in hand_landmarks.landmark]

                    x_min, x_max = max(min(x_coords) - 20, 0), min(max(x_coords) + 20, w)
                    y_min, y_max = max(min(y_coords) - 20, 0), min(max(y_coords) + 20, h)

                    hand_img = frame[y_min:y_max, x_min:x_max]
                    if hand_img.size == 0:
                        continue

                    hand_img = cv2.resize(hand_img, (128, 128))
                    img_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".jpg"
                    img_path = os.path.join(dataset_folder, label, img_name)
                    cv2.imwrite(img_path, hand_img)
                    saved += 1

                    cv2.putText(frame, f"{label}: {saved}/{num_images}", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    cv2.imshow("Dataset Collector", frame)

            print(f"✅ Done collecting {num_images} images for '{label}'\n")

cap.release()
cv2.destroyAllWindows()
