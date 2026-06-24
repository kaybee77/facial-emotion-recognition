"""
7-class image classifier — PyTorch transfer-learning pipeline.

This is a Python rebuild of a MATLAB Deep Learning Toolbox workflow that was
reverse-engineered from a saved workspace (BESTCOMBINED.mat). The MATLAB version
used:
  - imageDatastore (one folder per class) with train/val/test split
  - augmentedImageDatastore resizing images to 224x224x3
  - transfer learning on a pretrained CNN (SeriesNetwork / DAGNetwork) with the
    final fullyConnectedLayer replaced for 7 classes
  - trainingOptions('sgdm', ...)  -> SGD with momentum
  - class balancing (desiredNumObservationsPerClass / MinCount)
  - confusionmat + confusionchart and an overall accuracy score (~0.94)

Expected data layout (ImageFolder convention — one subfolder per class):

    data_root/
        class1/  img001.jpg ...
        class2/  ...
        ...
        class7/  ...

Usage:
    python train_classifier.py --data-root "H:/Desktop/final datasets/combined/unsharpcombine"
    python train_classifier.py --data-root ./data --arch resnet50 --epochs 30
"""

import argparse
import copy
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from torchvision import datasets, models, transforms

# scikit-learn is only needed for the confusion matrix / report at the end.
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# ImageNet normalization stats — required because we use pretrained weights.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
INPUT_SIZE = 224  # matches the MATLAB inputSize = [224 224 3]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def build_transforms():
    """Train transforms include light augmentation; val/test are deterministic.

    Mirrors MATLAB's augmentedImageDatastore (resize to 224x224x3) plus the kind
    of random flips/rotations commonly attached to the training datastore.
    """
    train_tf = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


def split_dataset(data_root, train_frac, val_frac, seed):
    """Equivalent of MATLAB splitEachLabel: split one ImageFolder into 3 subsets.

    The validation/test subsets get deterministic (eval) transforms by wrapping
    the base ImageFolder twice — one instance per transform.
    """
    train_tf, eval_tf = build_transforms()
    full_train = datasets.ImageFolder(data_root, transform=train_tf)
    full_eval = datasets.ImageFolder(data_root, transform=eval_tf)

    n = len(full_train)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    n_test = n - n_train - n_val

    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n, generator=g).tolist()
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    train_ds = torch.utils.data.Subset(full_train, train_idx)
    val_ds = torch.utils.data.Subset(full_eval, val_idx)
    test_ds = torch.utils.data.Subset(full_eval, test_idx)
    return train_ds, val_ds, test_ds, full_train.classes, full_train.targets


def make_balanced_sampler(train_subset, all_targets):
    """Class balancing — the MATLAB script equalized observations per class.

    A WeightedRandomSampler oversamples minority classes so each batch is roughly
    balanced, achieving the same effect as desiredNumObservationsPerClass without
    discarding data.
    """
    train_targets = [all_targets[i] for i in train_subset.indices]
    counts = Counter(train_targets)
    class_weights = {c: 1.0 / n for c, n in counts.items()}
    sample_weights = [class_weights[t] for t in train_targets]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def build_model(arch, num_classes, freeze_backbone):
    """Transfer learning: load a pretrained CNN and swap the classifier head.

    arch="resnet50" mirrors the MATLAB DAGNetwork (trantwk); "alexnet" mirrors
    the SeriesNetwork (net). Replacing the head == replacing fullyConnectedLayer.
    """
    if arch == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        head_params = model.fc.parameters()
    elif arch == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        head_params = model.fc.parameters()
    elif arch == "alexnet":
        model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)
        head_params = model.classifier[6].parameters()
    else:
        raise ValueError(f"Unsupported arch: {arch}")

    if freeze_backbone:
        # Feature-extraction mode: freeze everything, then re-enable the new head.
        for p in model.parameters():
            p.requires_grad = False
        for p in head_params:
            p.requires_grad = True
    return model


# --------------------------------------------------------------------------- #
# Train / evaluate
# --------------------------------------------------------------------------- #
def run_epoch(model, loader, criterion, optimizer, device, train):
    model.train() if train else model.eval()
    running_loss, running_correct, total = 0.0, 0, 0
    torch.set_grad_enabled(train)
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if train:
            optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        if train:
            loss.backward()
            optimizer.step()
        running_loss += loss.item() * images.size(0)
        running_correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    torch.set_grad_enabled(True)
    return running_loss / total, running_correct / total


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    for images, labels in loader:
        images = images.to(device)
        preds = model(images).argmax(1).cpu().numpy()
        y_pred.extend(preds.tolist())
        y_true.extend(labels.numpy().tolist())
    return np.array(y_true), np.array(y_pred)


def main():
    ap = argparse.ArgumentParser(description="7-class transfer-learning classifier (MATLAB->PyTorch)")
    ap.add_argument("--data-root", required=True, help="Folder with one subfolder per class")
    ap.add_argument("--arch", default="resnet50", choices=["resnet50", "resnet18", "alexnet"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)         # MATLAB InitialLearnRate
    ap.add_argument("--momentum", type=float, default=0.9)     # 'sgdm'
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--freeze-backbone", action="store_true",
                    help="Feature-extraction mode (train only the new head)")
    ap.add_argument("--no-balance", action="store_true", help="Disable class balancing")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="best_model.pt")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_root = Path(args.data_root)
    if not data_root.is_dir():
        raise SystemExit(f"data-root not found: {data_root}")

    train_ds, val_ds, test_ds, classes, all_targets = split_dataset(
        str(data_root), args.train_frac, args.val_frac, args.seed
    )
    num_classes = len(classes)
    print(f"Classes ({num_classes}): {classes}")
    print(f"Split -> train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    if args.no_balance:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.workers, pin_memory=True)
    else:
        sampler = make_balanced_sampler(train_ds, all_targets)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                                  num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True)

    model = build_model(args.arch, num_classes, args.freeze_backbone).to(device)
    criterion = nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_val_acc, best_state = 0.0, copy.deepcopy(model.state_dict())
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step()
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f}")
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state = copy.deepcopy(model.state_dict())

    # Restore best weights (equivalent to OutputNetwork='best-validation-loss').
    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "classes": classes, "arch": args.arch}, args.out)
    print(f"\nBest val accuracy: {best_val_acc:.4f}  ->  saved {args.out}")

    # ----- Final evaluation on the held-out test set -----
    y_true, y_pred = predict(model, test_loader, device)
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nTest accuracy: {acc:.4f}")
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(cm)
    print("\nPer-class report:")
    print(classification_report(y_true, y_pred, target_names=classes, digits=4))

    try:
        import matplotlib.pyplot as plt
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        disp.plot(xticks_rotation=45, cmap="Blues")
        plt.tight_layout()
        plt.savefig("confusion_matrix.png", dpi=150)
        print("Saved confusion_matrix.png")
    except Exception as e:
        print(f"(Skipped confusion-matrix plot: {e})")


if __name__ == "__main__":
    main()
