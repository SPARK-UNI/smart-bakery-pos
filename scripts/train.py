# src/train.py
import os
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# ----------------------------
# Kiểm tra GPU
# ----------------------------
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        print(f"✅ Found {len(gpus)} GPU(s):")
        for gpu in gpus:
            print("  •", gpu)
        # Tùy chọn: Giới hạn TensorFlow chỉ dùng 1 GPU (nếu bạn có nhiều)
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("⚙️  TensorFlow is configured to use GPU for training.")
    except RuntimeError as e:
        print("⚠️ RuntimeError khi cấu hình GPU:", e)
else:
    print("❌ Không phát hiện GPU. TensorFlow sẽ dùng CPU để train.")

# ----------------------------
# Cấu hình chung
# ----------------------------
IMG_SIZE = (224, 224)           # khớp với app.py
BATCH_SIZE = 32
EPOCHS = 20
SEED = 42

# Thư mục dữ liệu (sửa nếu bạn dùng tên khác)
DATA_DIR = Path("data")
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR   = DATA_DIR / "valid"
TEST_DIR  = DATA_DIR / "test"

# Thư mục lưu model/labels
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODELS_DIR / "bakery_cnn.h5"
LABELS_PATH = MODELS_DIR / "labels.txt"

# ----------------------------
# Data generators + Augment
# ----------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=12,
    width_shift_range=0.08,
    height_shift_range=0.08,
    shear_range=0.08,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR.as_posix(),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    seed=SEED
)

val_gen = val_datagen.flow_from_directory(
    VAL_DIR.as_posix(),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    seed=SEED
)

num_classes = train_gen.num_classes

# ----------------------------
# Kiến trúc CNN “tự viết” (không dùng pretrained)
# ----------------------------
def build_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), num_classes=3):
    inputs = layers.Input(shape=input_shape)

    # Block 1
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    # Block 2
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    # Block 3
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.3)(x)

    # Head
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

model = build_model(num_classes=num_classes)
model.summary()

# ----------------------------
# Callbacks
# ----------------------------
early_stop = callbacks.EarlyStopping(
    monitor="val_accuracy", patience=5, mode="max", restore_best_weights=True
)
ckpt = callbacks.ModelCheckpoint(
    MODEL_PATH.as_posix(),
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1
)

# ----------------------------
# Train
# ----------------------------
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[early_stop, ckpt]
)

# Nếu ModelCheckpoint chưa lưu (ví dụ không cải thiện), ta vẫn lưu bản cuối:
if not MODEL_PATH.exists():
    model.save(MODEL_PATH.as_posix())

# ----------------------------
# Ghi nhãn ra labels.txt (mỗi dòng là 1 lớp, đúng thứ tự chỉ số)
# ----------------------------
# class_indices: {'ten_lop': idx}
idx_to_class = {v: k for k, v in train_gen.class_indices.items()}
labels = [idx_to_class[i] for i in range(num_classes)]
with open(LABELS_PATH, "w", encoding="utf-8") as f:
    for name in labels:
        f.write(f"{name}\n")

print(f"Saved model to  : {MODEL_PATH}")
print(f"Saved labels to : {LABELS_PATH}")

# ----------------------------
# (Tùy chọn) Đánh giá nhanh trên test set nếu có
# ----------------------------
if TEST_DIR.exists() and any(TEST_DIR.iterdir()):
    test_gen = val_datagen.flow_from_directory(
        TEST_DIR.as_posix(),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )
    loss, acc = model.evaluate(test_gen)
    print(f"Test accuracy: {acc:.4f}")
