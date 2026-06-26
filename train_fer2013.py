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

FER2013 is heavily imbalanced (happiness ~25% vs disgust ~1.5%). To counter this:
    --class-weights      reweight the loss so rare classes count more
    --loss focal         use focal loss (focuses on hard/minority examples)
    --monitor val_macro_f1   select/checkpoint the model on macro-F1 (balance-aware)
                             instead of plain accuracy (majority-biased)
Macro-F1 and balanced accuracy are always reported on the test set.

Usage:
    python train_fer2013.py                                   # baseline, 25 epochs
    python train_fer2013.py --class-weights                   # recommended quick win
    python train_fer2013.py --class-weights --loss focal      # + focal loss
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
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
)

import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.callbacks import (
    Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint)
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


class MacroF1(Callback):
    """Compute validation macro-F1 each epoch and log it as 'val_macro_f1'.

    Plain accuracy is dominated by the majority classes, so monitoring macro-F1
    (the unweighted mean of per-class F1) makes checkpointing / early-stopping
    balance-aware. Must be placed FIRST in the callbacks list so the metric is
    available to the ModelCheckpoint/EarlyStopping callbacks that follow.
    """

    def __init__(self, X_val, y_val, batch_size=32):
        super().__init__()
        self.X_val = X_val
        self.y_true = y_val.argmax(1)
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs is not None else {}
        # Manual batched inference via __call__ instead of model.predict() —
        # repeated predict() calls leak memory (retracing), which crashed a run
        # at epoch 24 with "MemoryError: bad allocation".
        preds = []
        for i in range(0, len(self.X_val), self.batch_size):
            batch = self.X_val[i:i + self.batch_size]
            preds.append(np.asarray(self.model(batch, training=False)).argmax(1))
        y_pred = np.concatenate(preds)
        logs["val_macro_f1"] = f1_score(self.y_true, y_pred, average="macro")
        print(f"  - val_macro_f1: {logs['val_macro_f1']:.4f}")


def main():
    ap = argparse.ArgumentParser(description="VGG-19 on FER2013 (70/15/15 split)")
    ap.add_argument("--csv", default="fer2013.csv")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="best_model_fer.keras")
    # ---- imbalance-correction options ----
    ap.add_argument("--class-weights", action="store_true",
                    help="Reweight the loss with balanced class weights")
    ap.add_argument("--loss", choices=["ce", "focal"], default="ce",
                    help="ce = categorical cross-entropy, focal = focal loss")
    ap.add_argument("--focal-gamma", type=float, default=2.0,
                    help="Focusing parameter for focal loss")
    ap.add_argument("--monitor", choices=["val_accuracy", "val_macro_f1"],
                    default="val_macro_f1",
                    help="Metric used for checkpoint / early-stop / LR schedule")
    ap.add_argument("--weight-scheme", choices=["balanced", "sqrt"], default="sqrt",
                    help="Class-weight scaling: full 'balanced' (aggressive, can "
                         "diverge) or softened 'sqrt' (recommended, stable)")
    ap.add_argument("--clipnorm", type=float, default=1.0,
                    help="Gradient-clipping norm for stability (0 disables)")
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
    n_total = len(feats)
    print(f"Split -> train {len(X_train)} | val {len(X_val)} | test {len(X_test)} "
          f"({len(X_train)/n_total:.0%}/{len(X_val)/n_total:.0%}/{len(X_test)/n_total:.0%})")
    # Free the ~1 GB full-dataset array; the split copies are all we need now.
    import gc
    del feats
    gc.collect()

    # ---- loss: cross-entropy or focal (focal down-weights easy/majority) ----
    if args.loss == "focal":
        loss = tf.keras.losses.CategoricalFocalCrossentropy(gamma=args.focal_gamma)
    else:
        loss = "categorical_crossentropy"

    # Gradient clipping stabilizes training (prevents the divergence seen with
    # aggressive class weights on a fully fine-tuned VGG-19).
    opt_kwargs = {"learning_rate": args.lr}
    if args.clipnorm and args.clipnorm > 0:
        opt_kwargs["clipnorm"] = args.clipnorm

    model = build_vgg19(y.shape[1])
    model.compile(loss=loss,
                  optimizer=optimizers.Adam(**opt_kwargs),
                  metrics=["accuracy"])

    # ---- class weights: balanced -> rare classes (disgust) weigh more ----
    class_weight = None
    if args.class_weights:
        cls = np.arange(y.shape[1])
        weights = compute_class_weight("balanced", classes=cls, y=y_train.argmax(1))
        if args.weight_scheme == "sqrt":
            # Soften extreme weights (disgust ~9.4x -> ~3x) and renormalize to
            # mean 1 so the effective learning rate stays stable.
            weights = np.sqrt(weights)
            weights = weights / weights.mean()
        class_weight = dict(zip(cls.tolist(), weights.tolist()))
        print(f"Class weights ({args.weight_scheme}):",
              {EMOTION_NAMES.get(int(le.classes_[c]), int(c)): round(w, 2)
               for c, w in class_weight.items()})

    train_datagen = ImageDataGenerator(
        rotation_range=15, width_shift_range=0.15, height_shift_range=0.15,
        shear_range=0.15, zoom_range=0.15, horizontal_flip=True)
    train_datagen.fit(X_train)

    # MacroF1 must come first so the metric exists for the callbacks below.
    mode = "max"
    callbacks = [
        MacroF1(X_val, y_val, batch_size=args.batch_size),
        ModelCheckpoint(args.out, monitor=args.monitor, mode=mode,
                        save_best_only=True, verbose=1),
        EarlyStopping(monitor=args.monitor, mode=mode, min_delta=5e-5,
                      patience=11, verbose=1, restore_best_weights=True),
        ReduceLROnPlateau(monitor=args.monitor, mode=mode, factor=0.5,
                          patience=7, min_lr=1e-7, verbose=1),
    ]
    print(f"Loss: {args.loss} | class-weights: {bool(args.class_weights)} "
          f"({args.weight_scheme}) | clipnorm: {args.clipnorm} | "
          f"model-selection metric: {args.monitor}")

    history = model.fit(
        train_datagen.flow(X_train, y_train, batch_size=args.batch_size),
        validation_data=(X_val, y_val),
        steps_per_epoch=len(X_train) // args.batch_size,
        epochs=args.epochs, callbacks=callbacks, class_weight=class_weight,
        verbose=2)

    with open("fer_classes.json", "w") as f:
        json.dump([EMOTION_NAMES.get(int(c), int(c)) for c in le.classes_], f)

    # ---- Evaluate on the held-out 15% test set ----
    y_pred = model.predict(X_test, batch_size=args.batch_size).argmax(1)
    y_true = y_test.argmax(1)
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    names = [EMOTION_NAMES.get(int(c), int(c)) for c in le.classes_]
    print(f"\n==== TEST accuracy: {acc:.4f} | balanced accuracy: {bal_acc:.4f} "
          f"| macro-F1: {macro_f1:.4f} ====")
    print("(balanced accuracy & macro-F1 weight every class equally — the metrics "
          "to watch under imbalance)")
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=names, digits=4))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        h = history.history
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(h["accuracy"], label="train acc"); ax[0].plot(h["val_accuracy"], label="val acc")
        if "val_macro_f1" in h:
            ax[0].plot(h["val_macro_f1"], label="val macro-F1", linestyle="--")
        ax[0].set_title("Accuracy / Macro-F1"); ax[0].legend()
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
