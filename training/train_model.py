"""
train_model.py
---------------
Trains a MobileNetV2-based classifier for diabetic retinopathy severity
(5 classes: No DR, Mild, Moderate, Severe, Proliferative DR).

DATASET:
Download the APTOS 2019 Blindness Detection dataset from Kaggle:
    https://www.kaggle.com/competitions/aptos2019-blindness-detection/data
(Free Kaggle account required. ~10k labeled retinal images.)

After downloading, arrange the data like this:

    data/
        train_images/          <- all .png images from train_images.zip
        train.csv              <- columns: id_code, diagnosis (0-4)

Then run:
    python train_model.py --data_dir ./data --epochs 20

This will produce model/dr_mobilenetv2.h5, ready for the Flask app.
"""

import os
import argparse
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

IMG_SIZE = 224
NUM_CLASSES = 5


def preprocess_image(path, img_size=IMG_SIZE):
    """Load an image, crop to a centered square, resize, apply CLAHE
    contrast enhancement (helps highlight retinal vessels/lesions)."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Center-crop to square based on the shorter side
    h, w, _ = img.shape
    side = min(h, w)
    top = (h - side) // 2
    left = (w - side) // 2
    img = img[top:top + side, left:left + side]

    img = cv2.resize(img, (img_size, img_size))

    # CLAHE on the L channel (LAB color space) improves lesion visibility
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return img


def build_dataset(df, images_dir, img_size=IMG_SIZE, batch_size=32, training=True):
    paths = [os.path.join(images_dir, f"{i}.png") for i in df["id_code"]]
    labels = df["diagnosis"].values

    def _load(path, label):
        path = path.numpy().decode("utf-8")
        img = preprocess_image(path, img_size)
        img = preprocess_input(img.astype(np.float32))
        return img, label

    def _wrap(path, label):
        img, label = tf.py_function(_load, [path, label], [tf.float32, tf.int64])
        img.set_shape([img_size, img_size, 3])
        label.set_shape([])
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(buffer_size=len(paths))
    ds = ds.map(_wrap, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def build_model(img_size=IMG_SIZE, num_classes=NUM_CLASSES, fine_tune_at=100):
    base_model = MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = True
    # Freeze the earlier layers; fine-tune the later ones
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    inputs = layers.Input(shape=(img_size, img_size, 3))
    x = base_model(inputs, training=True)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output", type=str, default="../model/dr_mobilenetv2.h5")
    args = parser.parse_args()

    csv_path = os.path.join(args.data_dir, "train.csv")
    images_dir = os.path.join(args.data_dir, "train_images")

    df = pd.read_csv(csv_path)
    train_df, val_df = train_test_split(
        df, test_size=0.15, stratify=df["diagnosis"], random_state=42
    )

    print(f"Train: {len(train_df)}  Val: {len(val_df)}")

    train_ds = build_dataset(train_df, images_dir, batch_size=args.batch_size, training=True)
    val_ds = build_dataset(val_df, images_dir, batch_size=args.batch_size, training=False)

    # Handle class imbalance (No DR is usually overrepresented)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_df["diagnosis"]),
        y=train_df["diagnosis"],
    )
    class_weight_dict = dict(enumerate(class_weights))
    print("Class weights:", class_weight_dict)

    model = build_model()
    model.summary()

    cbs = [
        callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
        callbacks.ModelCheckpoint(args.output, monitor="val_accuracy", save_best_only=True),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weight_dict,
        callbacks=cbs,
    )

    model.save(args.output)
    print(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()
