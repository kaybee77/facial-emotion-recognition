"""
Train a VGG-19 facial-emotion classifier on FER2013.

Split: 70% train / 15% validation / 15% test (stratified by emotion).
Architecture matches the notebook: VGG19 (ImageNet weights, no top) ->
GlobalAveragePooling2D -> Dense(7, softmax). Inputs are 48x48 grayscale faces
expanded to 3 channels and scaled to [0, 1].

Outputs:
    best_model_fer.keras   - best-validation model (loadable for inference)
    fer_classes.json       - class label order
    training_curves.png    - accuracy/loss curves
    confusion_matrix.png   - test-set confusion matrix

Usage:
    python train_fer2013.py                       # defaults: fer2013.csv, 25 epochs
    python train_fer2013.py --epochs 40 --batch-size 64
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import cv2

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
)

import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# FER2013 emotion code -> name
EMOTION_NAMES = {0: "anger", 1: "disgust", 2: "fear", 3: "happiness",
                 4: "sadness", 5: "surprise", 6: "neutral"}


def load_fer2013(csv_path):
    df = pd.read_csv(csv_path)
    # 48x48 grayscale -> 3-channel (VGG19 expects 3 channels)
    img = df.pixels.apply(lambda s: np.array(s.split(), dtype="float32").reshape(48, 48))
    img = np.stack(img.to_numpy(), axis=0)
    feats = np.stack([cv2.cvtColor(x, cv2.COLOR_GRAY2RGB) for x in img], axis=0)
    feats /= 255.0
    le = LabelEncoder()
    y_int = le.fit_transform(df.emotion)
    y = tf.keras.utils.to_categorical(y_int)
    return feats, y, y_int, le


def stratified_70_15_15(feats, y, y_int, seed):
    """70% train, 15% val, 15% test — stratified on emotion."""
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        feats, y, test_size=0.30, stratify=y_int, random_state=seed)
    strat_tmp = y_tmp.argmax(1)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=strat_tmp, random_state=seed)
    return X_train, y_train, X_val, y_val, X_test, y_test


def build_vgg19(num_classes):
    vgg = tf.keras.applications.VGG19(
        weights="imagenet", include_top=False, input_shape=(48, 48, 3))
    x = vgg.layers[-2].output            # drop final maxpool (as in the notebook)
    x = GlobalAveragePooling2D()(x)
    out = Dense(num_classes, activation="softmax", name="out_layer")(x)
    return Model(inputs=vgg.input, outputs=out)


def main():
    ap = argparse.ArgumentParser(description="VGG-19 on FER2013 (70/15/15 split)")
    ap.add_argument("--csv", default="fer2013.csv")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="best_model_fer.keras")
    args = ap.parse_args()

    if not Path(args.csv).exists():
        raise SystemExit(f"CSV not found: {args.csv}")

    print(f"TensorFlow {tf.__version__} | GPUs: {tf.config.list_physical_devices('GPU')}")
    print("Loading FER2013 ...")
    feats, y, y_int, le = load_fer2013(args.csv)
    print(f"Data: {feats.shape}, classes (in label order): "
          f"{[EMOTION_NAMES.get(int(c), int(c)) for c in le.classes_]}")

    X_train, y_train, X_val, y_val, X_test, y_test = stratified_70_15_15(
        feats, y, y_int, args.seed)
    print(f"Split -> train {len(X_train)} | val {len(X_val)} | test {len(X_test)} "
          f"({len(X_train)/len(feats):.0%}/{len(X_val)/len(feats):.0%}/{len(X_test)/len(feats):.0%})")

    model = build_vgg19(y.shape[1])
    model.compile(loss="categorical_crossentropy",
                  optimizer=optimizers.Adam(learning_rate=args.lr),
                  metrics=["accuracy"])

    train_datagen = ImageDataGenerator(
        rotation_range=15, width_shift_range=0.15, height_shift_range=0.15,
        shear_range=0.15, zoom_range=0.15, horizontal_flip=True)
    train_datagen.fit(X_train)

    callbacks = [
        ModelCheckpoint(args.out, monitor="val_accuracy", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_accuracy", min_delta=5e-5, patience=11,
                      verbose=1, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=7,
                          min_lr=1e-7, verbose=1),
    ]

    history = model.fit(
        train_datagen.flow(X_train, y_train, batch_size=args.batch_size),
        validation_data=(X_val, y_val),
        steps_per_epoch=len(X_train) // args.batch_size,
        epochs=args.epochs, callbacks=callbacks, verbose=2)

    with open("fer_classes.json", "w") as f:
        json.dump([EMOTION_NAMES.get(int(c), int(c)) for c in le.classes_], f)

    # ---- Evaluate on the held-out 15% test set ----
    y_pred = model.predict(X_test, batch_size=args.batch_size).argmax(1)
    y_true = y_test.argmax(1)
    acc = accuracy_score(y_true, y_pred)
    names = [EMOTION_NAMES.get(int(c), int(c)) for c in le.classes_]
    print(f"\n==== TEST accuracy: {acc:.4f} ====")
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=names, digits=4))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        h = history.history
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(h["accuracy"], label="train"); ax[0].plot(h["val_accuracy"], label="val")
        ax[0].set_title("Accuracy"); ax[0].legend()
        ax[1].plot(h["loss"], label="train"); ax[1].plot(h["val_loss"], label="val")
        ax[1].set_title("Loss"); ax[1].legend()
        fig.tight_layout(); fig.savefig("training_curves.png", dpi=150)
        ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred), display_labels=names)\
            .plot(cmap="Blues", xticks_rotation=45)
        plt.tight_layout(); plt.savefig("confusion_matrix.png", dpi=150)
        print("Saved training_curves.png and confusion_matrix.png")
    except Exception as e:
        print(f"(plot skipped: {e})")

    print(f"Saved best model -> {args.out}")


if __name__ == "__main__":
    main()
