# ASL-Alphabet-Detection
This project performs real-time American Sign Language (ASL) alphabet recognition using MediaPipe hand landmarks and a deep learning model. It supports static signs including A–Z, space, delete, and nothing. To ensure consistency across varied hand sizes and positions, all landmarks are normalized relative to the wrist and scaled uniformly. The system works with both webcam and video input, and the trained model can be exported to TensorFlow Lite for deployment on edge or mobile devices.

✨ Features
🔤 Recognizes ASL alphabets: A–Z, space, delete, nothing

🎥 Real-time webcam and video file input

✋ Extracts 21 hand landmarks using MediaPipe

📐 Wrist-relative and scale-invariant normalization

🧠 Trained model with 94%+ validation accuracy

📦 TensorFlow Lite export supported

📁 Main Files
File	Purpose
real_time_webcam.py	Run real-time prediction with webcam
extract_landmarks.py	Extract and normalize training data
train_combined_model.py	Train the deep learning model
label_map.json	Class-to-label index mapping
asl_wlasl_landmark_model_improved.h5	Final trained model
tflite_export.py (opt)	Export model to TensorFlow Lite

🚀 Getting Started
Install dependencies:
pip install opencv-python mediapipe tensorflow

Run real-time recognition:
python real_time_webcam.py

Retrain the model:
python extract_landmarks.py
python train_combined_model.py

(Optional) Export to TensorFlow Lite:
python tflite_export.py
