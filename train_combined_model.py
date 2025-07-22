import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, models
import json

# === Load and Clean Data ===
df = pd.read_csv("combined_asl_wlasl_landmarks.csv")
df = df[df.iloc[:, 1:].sum(axis=1) != 0]  # Remove empty/invalid rows
print(f"✅ Using full dataset with {len(df)} samples")

# === Features and Labels ===
X = df.drop('label', axis=1).values.astype('float32')
y_raw = df['label'].values

# === Optional: Gaussian Noise Augmentation ===
def add_noise(X, noise_level=0.01):
    noise = np.random.normal(0, noise_level, X.shape)
    return np.vstack((X, X + noise))

X = add_noise(X)
y_raw = np.concatenate([y_raw, y_raw])

# === Encode Labels ===
encoder = LabelEncoder()
y = encoder.fit_transform(y_raw)
num_classes = len(encoder.classes_)

# === Save Label Map ===
with open("label_map_combined.json", "w") as f:
    json.dump({i: label for i, label in enumerate(encoder.classes_)}, f)
print(f"✅ Saved label map with {num_classes} classes")

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# === Model Definition ===
model = models.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
])

# === Compile ===
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# === Callbacks ===
early_stop = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, verbose=1)

# === Train ===
model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=64,
    validation_data=(X_test, y_test),
    callbacks=[early_stop, lr_schedule]
)

# === Save Model ===
model.save("asl_wlasl_landmark_model_improved.h5")
print("✅ Model saved as 'asl_wlasl_landmark_model_improved.h5'")
