"""
7-class image classifier — TensorFlow/Keras transfer-learning pipeline.

Keras rebuild of the MATLAB Deep Learning Toolbox workflow reverse-engineered
from BESTCOMBINED.mat. MATLAB -> Keras mapping:
  imageDatastore (folder per class)      -> image_dataset_from_directory
  separate train/val/test datastores     -> data/train, data/val, data/test
  augmentedImageDatastore (224x224x3)     -> Resizing + RandomFlip/Rotation layers
  resnet50 / DAGNetwork + transfer learn  -> applications.ResNet50(include_top=False)
  fullyConnectedLayer(7)                  -> Dense(7, activation='softmax')
  trainingOptions('sgdm', ...)            -> optimizers.SGD(momentum=0.9)
  desiredNumObservationsPerClass (balance)-> class_weight in model.fit
  confusionmat / confusionchart           -> sklearn confusion_matrix + display
  classify / accuracy                     -> model.predict + accuracy_score

Expected data layout (PRE-SPLIT — one subfolder per class inside each split):

    data_root/
        train/  class1/ ...  class2/ ...  ... class7/ ...
        val/    class1/ ...                    class7/ ...
        test/   class1/ ...                    class7/ ...

Usage:
    python train_classifier_keras.py --data-root ./data
    python train_classifier_keras.py --data-root ./data --arch resnet50 --epochs 30
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

INPUT_SIZE = 224  # matches the MATLAB inputSize = [224 224 3]

# Each backbone ships its own preprocessing (scaling/mean-subtraction).
ARCHES = {
    "resnet50": (keras.applications.ResNet50,
                 keras.applications.resnet50.preprocess_input),
    "resnet101": (keras.applications.ResNet101,
                  keras.applications.resnet.preprocess_input),
    "vgg16": (keras.applications.VGG16,          # closest to the MATLAB SeriesNetwork
              keras.applications.vgg16.preprocess_input),
}


def make_dataset(split_dir, batch_size, shuffle, seed):
    """One image_dataset_from_directory per split (= one imageDatastore)."""
    ds = keras.utils.image_dataset_from_directory(
        split_dir,
        labels="inferred",
        label_mode="int",
        image_size=(INPUT_SIZE, INPUT_SIZE),
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
    )
    return ds


def augmentation_block():
    """Training-time augmentation, mirroring the augmented training datastore."""
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
    ], name="augmentation")


def build_model(arch, num_classes, freeze_backbone, lr, momentum, weight_decay):
    base_cls, preprocess = ARCHES[arch]
    base = base_cls(include_top=False, weights="imagenet",
                    input_shape=(INPUT_SIZE, INPUT_SIZE, 3), pooling="avg")
    base.trainable = not freeze_backbone  # freeze == feature-extraction mode

    inputs = keras.Input(shape=(INPUT_SIZE, INPUT_SIZE, 3))
    x = augmentation_block()(inputs)
    x = preprocess(x)                       # backbone-specific normalization
    x = base(x, training=not freeze_backbone)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)  # = fullyConnectedLayer(7)
    model = keras.Model(inputs, outputs)

    optimizer = keras.optimizers.SGD(            # 'sgdm'
        learning_rate=lr, momentum=momentum, weight_decay=weight_decay)
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def compute_class_weights(train_dir, class_names):
    """class_weight dict for balancing (= desiredNumObservationsPerClass)."""
    counts = Counter()
    for idx, name in enumerate(class_names):
        counts[idx] = len(list((Path(train_dir) / name).glob("*")))
    total = sum(counts.values())
    n = len(class_names)
    return {c: total / (n * cnt) for c, cnt in counts.items() if cnt > 0}


def gather_labels(ds):
    """Pull integer labels out of a tf.data pipeline (for the confusion matrix)."""
    return np.concatenate([y.numpy() for _, y in ds], axis=0)


def main():
    ap = argparse.ArgumentParser(description="7-class transfer-learning classifier (MATLAB->Keras)")
    ap.add_argument("--data-root", required=True,
                    help="Folder containing train/ val/ test/ subfolders")
    ap.add_argument("--arch", default="resnet50", choices=list(ARCHES))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)        # InitialLearnRate
    ap.add_argument("--momentum", type=float, default=0.9)    # 'sgdm'
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--freeze-backbone", action="store_true",
                    help="Feature-extraction mode (train only the new head)")
    ap.add_argument("--no-balance", action="store_true", help="Disable class balancing")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="best_model.keras")
    args = ap.parse_args()

    keras.utils.set_random_seed(args.seed)

    root = Path(args.data_root)
    train_dir, val_dir, test_dir = root / "train", root / "val", root / "test"
    for d in (train_dir, val_dir, test_dir):
        if not d.is_dir():
            raise SystemExit(f"Missing split folder: {d}\n"
                             f"Expected {root}/train, {root}/val, {root}/test")

    train_ds = make_dataset(str(train_dir), args.batch_size, shuffle=True, seed=args.seed)
    val_ds = make_dataset(str(val_dir), args.batch_size, shuffle=False, seed=args.seed)
    test_ds = make_dataset(str(test_dir), args.batch_size, shuffle=False, seed=args.seed)

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")

    # Cache/prefetch for throughput (does not change results).
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)

    model = build_model(args.arch, num_classes, args.freeze_backbone,
                        args.lr, args.momentum, args.weight_decay)
    model.summary()

    class_weight = None if args.no_balance else compute_class_weights(train_dir, class_names)
    if class_weight:
        print(f"Class weights: {class_weight}")

    callbacks = [
        # Keep best-validation weights (= OutputNetwork='best-validation-loss').
        keras.callbacks.ModelCheckpoint(args.out, monitor="val_accuracy",
                                        save_best_only=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.1,
                                          patience=5, verbose=1),
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
              class_weight=class_weight, callbacks=callbacks)

    # ----- Final evaluation on the held-out test set -----
    best = keras.models.load_model(args.out)
    y_true = gather_labels(test_ds)
    probs = best.predict(test_ds)
    y_pred = probs.argmax(axis=1)

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nTest accuracy: {acc:.4f}")
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(cm)
    print("\nPer-class report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    print(f"\nSaved best model -> {args.out}")

    try:
        import matplotlib.pyplot as plt
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(xticks_rotation=45, cmap="Blues")
        plt.tight_layout()
        plt.savefig("confusion_matrix.png", dpi=150)
        print("Saved confusion_matrix.png")
    except Exception as e:
        print(f"(Skipped confusion-matrix plot: {e})")


if __name__ == "__main__":
    main()
