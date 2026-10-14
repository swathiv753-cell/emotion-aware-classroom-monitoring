
from pathlib import Path
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "datasetemotion"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Settings
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
SEED = 42

tf.keras.utils.set_random_seed(SEED)

if not TRAIN_DIR.is_dir() or not TEST_DIR.is_dir():
    raise FileNotFoundError("Check datasetemotion/train and datasetemotion/test.")

# Split training data into training (85%) and validation (15%).
# The test set is kept separate until final evaluation.
train_ds = keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    color_mode="rgb",
    validation_split=0.15,
    subset="training",
    seed=SEED,
    shuffle=True
)

val_ds = keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    color_mode="rgb",
    validation_split=0.15,
    subset="validation",
    seed=SEED,
    shuffle=False
)

test_ds = keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    color_mode="rgb",
    shuffle=False
)

class_names = train_ds.class_names
print("Emotion classes:", class_names)

if len(class_names) != 7:
    raise ValueError(f"Expected 7 emotion classes, found {len(class_names)}")

if val_ds.class_names != class_names or test_ds.class_names != class_names:
    raise ValueError("Train, validation, and test class names must match.")

with open(MODEL_DIR / "class_names.json", "w") as f:
    json.dump(class_names, f)

# Balance classes so rare emotions contribute more during training.
class_counts = {
    name: sum(
        1 for p in (TRAIN_DIR / name).rglob("*")
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    for name in class_names
}

total_images = sum(class_counts.values())
num_classes = len(class_names)

class_weights = {
    i: total_images / (num_classes * class_counts[name])
    for i, name in enumerate(class_names)
    if class_counts[name] > 0
}

print("Training class counts:", class_counts)
print("Class weights:", class_weights)

# Performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# Augmentation applies only during training.
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
], name="augmentation")

# Pretrained feature extractor
base_model = keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

inputs = keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)

# MobileNetV2 input range: [-1, 1]
x = layers.Rescaling(1.0 / 127.5, offset=-1)(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.35)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

best_path = str(MODEL_DIR / "best_emotion_model.keras")

callbacks = [
    keras.callbacks.ModelCheckpoint(
        best_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )
]

# Phase 1: Train the classification head.
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)

# Evaluate only after training, on the untouched test set.
best_model = keras.models.load_model(best_path)
test_loss, test_accuracy = best_model.evaluate(test_ds, verbose=1)

print(f"\nTest loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy * 100:.2f}%")
print("Best model saved to:", best_path)
print("Class names saved to:", MODEL_DIR / "class_names.json")
