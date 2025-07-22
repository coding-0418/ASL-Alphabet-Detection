import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import json
from collections import deque, Counter

# === Load model and label map ===
model = tf.keras.models.load_model("asl_wlasl_landmark_model_improved.h5")
with open("label_map_combined.json", "r") as f:
    label_map = json.load(f)

# === MediaPipe setup ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# === Prediction smoothing buffer ===
prediction_buffer = deque(maxlen=15)

# === Robust normalization: wrist-relative + scale-invariant ===
def normalize_landmarks(landmarks):
    base = landmarks[0]
    translated = np.array([[lm.x - base.x, lm.y - base.y, lm.z - base.z] for lm in landmarks])
    max_value = np.max(np.abs(translated))
    if max_value > 0:
        translated /= max_value
    return translated.flatten().reshape(1, -1)

# === Webcam loop ===
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        landmarks = result.multi_hand_landmarks[0].landmark
        mp_drawing.draw_landmarks(frame, result.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

        input_data = normalize_landmarks(landmarks)
        prediction = model.predict(input_data, verbose=0)
        class_id = int(np.argmax(prediction))
        class_label = label_map[str(class_id)]
        prediction_buffer.append(class_label)

        smoothed = Counter(prediction_buffer).most_common(1)[0][0]
        cv2.putText(frame, f"Sign: {smoothed}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    cv2.imshow("ASL Webcam Inference", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
