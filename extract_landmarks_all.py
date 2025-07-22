import os
import cv2
import csv
import mediapipe as mp

# === Dataset directories ===
dataset_dirs = [
    "/home/dan/PycharmProjects/PythonProject/All/ASL2/asl_alphabet_train",
    "/home/dan/PycharmProjects/PythonProject/All/ASL1/asl_dataset",
    "/home/dan/PycharmProjects/PythonProject/All/WLASL/videos",
    "/home/dan/PycharmProjects/PythonProject/All/ASL3/dataset",
    "/home/dan/PycharmProjects/PythonProject/All/ASL4/datasets/asl_alphabet_train",
    "/home/dan/PycharmProjects/PythonProject/All/ASL5/asl-numbers-alphabet-dataset"
]

output_csv = "combined_asl_wlasl_landmarks.csv"
skipped_log = "skipped_files.txt"

# === MediaPipe Setup ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# === Extensions ===
IMAGE_EXTS = [".jpg", ".jpeg", ".png"]
VIDEO_EXTS = [".mp4", ".avi", ".mov"]

# === Resize + Pad for better detection ===
def resize_and_pad(img, size=256):
    h, w = img.shape[:2]
    scale = size / max(h, w)
    resized = cv2.resize(img, (int(w * scale), int(h * scale)))
    pad_top = (size - resized.shape[0]) // 2
    pad_bottom = size - resized.shape[0] - pad_top
    pad_left = (size - resized.shape[1]) // 2
    pad_right = size - resized.shape[1] - pad_left
    padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right,
                                borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return padded

# === Counters ===
count_total = 0
count_written = 0
count_no_hand = 0
count_unreadable = 0
count_unsupported = 0

# === Write CSV ===
with open(output_csv, mode='w', newline='') as f_out, open(skipped_log, 'w') as f_log:
    writer = csv.writer(f_out)
    header = ['label'] + [f'{axis}{i}' for i in range(21) for axis in ['x', 'y', 'z']]
    writer.writerow(header)

    for root_dir in dataset_dirs:
        print(f"🔍 Scanning: {root_dir}")
        for label in sorted(os.listdir(root_dir)):
            label_path = os.path.join(root_dir, label)
            if not os.path.isdir(label_path):
                continue

            for fname in os.listdir(label_path):
                fpath = os.path.join(label_path, fname)
                ext = os.path.splitext(fname)[-1].lower()
                count_total += 1

                # === Handle Images ===
                if ext in IMAGE_EXTS:
                    img = cv2.imread(fpath)
                    if img is None:
                        count_unreadable += 1
                        f_log.write(f"[Unreadable Image] {fpath}\n")
                        continue

                    img_prep = resize_and_pad(img)
                    img_rgb = cv2.cvtColor(img_prep, cv2.COLOR_BGR2RGB)
                    result = hands.process(img_rgb)

                    if result.multi_hand_landmarks:
                        landmarks = result.multi_hand_landmarks[0].landmark
                        row = [label] + [round(coord, 6) for lm in landmarks for coord in (lm.x, lm.y, lm.z)]
                        writer.writerow(row)
                        count_written += 1
                    else:
                        count_no_hand += 1
                        f_log.write(f"[No hand in image] {fpath}\n")

                # === Handle Videos ===
                elif ext in VIDEO_EXTS:
                    cap = cv2.VideoCapture(fpath)
                    if not cap.isOpened():
                        count_unreadable += 1
                        f_log.write(f"[Unreadable Video] {fpath}\n")
                        continue

                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frame_prep = resize_and_pad(frame)
                        frame_rgb = cv2.cvtColor(frame_prep, cv2.COLOR_BGR2RGB)
                        result = hands.process(frame_rgb)

                        if result.multi_hand_landmarks:
                            landmarks = result.multi_hand_landmarks[0].landmark
                            row = [label] + [round(coord, 6) for lm in landmarks for coord in (lm.x, lm.y, lm.z)]
                            writer.writerow(row)
                            count_written += 1
                        else:
                            count_no_hand += 1
                            f_log.write(f"[No hand in video frame] {fpath}\n")
                    cap.release()

                else:
                    count_unsupported += 1
                    f_log.write(f"[Unsupported File] {fpath}\n")

hands.close()

# === Summary ===
print("\n📊 Summary:")
print(f"✅ Total files scanned: {count_total}")
print(f"✅ Landmarks written: {count_written}")
print(f"❌ Skipped (no hand): {count_no_hand}")
print(f"❌ Skipped (unreadable): {count_unreadable}")
print(f"❌ Skipped (unsupported): {count_unsupported}")
print(f"📝 Log saved to: {skipped_log}")
print(f"📄 Output saved to: {output_csv}")
