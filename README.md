# Facial Emotion Recognition

A deep-learning facial-emotion classifier built in Python — trained on the
**FER2013** dataset with a **VGG-19** transfer-learning model, complete with
command-line tools, a real-time webcam demo, and a polished **Gradio web UI** for
live exhibitions.

The model recognizes 7 emotions: **anger, disgust, fear, happiness, sadness,
surprise, neutral**.

## Results

- VGG-19 (ImageNet weights) → GlobalAveragePooling → Dense(7, softmax).
- Stratified **70 / 15 / 15** train / validation / test split.
- A baseline reaches **67.1% test accuracy** (strong for FER2013 — human accuracy
  is ~65%, published SOTA ~73%).
- The shipped model uses **class weighting** to handle the imbalance (below),
  which raises minority-class performance at a small cost to raw accuracy.

| Metric | Baseline | Class-weighted (shipped) |
|--------|----------|--------------------------|
| Accuracy | 0.671 | 0.660 |
| **Balanced accuracy** | ~0.62 | **0.641** |
| Macro-F1 | 0.637 | 0.630 |
| **disgust recall** | 0.44 | **0.62** |

![Confusion matrix](confusion_matrix.png)

### Handling class imbalance

FER2013 is heavily imbalanced — `happiness` has ~25% of samples while `disgust`
has only ~1.5% (a 16× gap), which depresses minority-class recall.

![FER2013 class distribution](fer2013_distribution.png)

`train_fer2013.py` includes several correction techniques (off by default for the
baseline, enabled via flags):

- **`--class-weights`** — reweight the loss so rare classes count more
  (`--weight-scheme sqrt` softens the weights for stability; `balanced` is the
  full scheme).
- **`--loss focal`** — focal loss, which focuses on hard/minority examples.
- **`--monitor val_macro_f1`** — select/checkpoint on macro-F1 (balance-aware)
  instead of accuracy (majority-biased).
- **`--clipnorm`** — gradient clipping, needed to keep aggressive weighting stable.

**Balanced accuracy** and **macro-F1** are the metrics to watch under imbalance —
they weight every class equally. The class-weighted model improves disgust recall
from **0.44 → 0.62** and balanced accuracy from **~0.62 → 0.64**, with a small dip
in overall accuracy (the majority classes stop dominating — the intended
trade-off).

## Project layout

| File | Purpose |
|------|---------|
| `train_fer2013.py` | Train VGG-19 on FER2013 (70/15/15 split, imbalance options) |
| `evaluate.py` | Evaluate any saved model on the test split (per-class metrics) |
| `predict_fer.py` | Classify emotion in image files / folders (CLI) |
| `webcam_fer.py` | Real-time emotion recognition from a webcam (OpenCV) |
| `app.py` | **Gradio web UI** — webcam + upload, for live demos/exhibitions |
| `plot_distribution.py` | Bar charts of the FER2013 class distribution |
| `facial-emotion-recognition-vgg19-fer2013.ipynb` | Original notebook walkthrough |
| `train_classifier.py` | Alternative generic image classifier (PyTorch) |
| `train_classifier_keras.py` / `predict.py` | Alternative generic classifier (Keras) |

## Setup

```bash
pip install -r requirements.txt
```

> **Note:** TensorFlow is CPU-only on native Windows (≥ 2.11). Use WSL2 or a Linux
> machine with CUDA for GPU acceleration.

### Get the data / model (not in this repo — too large for GitHub)

- **`fer2013.csv`** — download from
  [Kaggle: FER2013](https://www.kaggle.com/datasets/msambare/fer2013) (or the
  Facial Expression Recognition Challenge) and place it in the project root.
- **`best_model_fer.keras`** — produced by running training (below).

## Usage

**Train the model:**
```bash
python train_fer2013.py --epochs 25 --batch-size 32          # baseline
python train_fer2013.py --class-weights                      # recommended (handles imbalance)
python train_fer2013.py --class-weights --loss focal         # + focal loss
```
Produces `best_model_fer.keras`, `fer_classes.json`, `confusion_matrix.png`,
and `training_curves.png`, and prints accuracy, balanced accuracy, and macro-F1.

**Evaluate a saved model on the test set:**
```bash
python evaluate.py --model best_model_fer.keras --cm-out confusion_matrix.png
```

**Classify images:**
```bash
python predict_fer.py --input my_photo.jpg
python predict_fer.py --input ./photos --topk 3 --detect   # detect+crop faces
```

**Real-time webcam (OpenCV window):**
```bash
python webcam_fer.py            # press q to quit, s to snapshot
```

**Exhibition web UI (Gradio):**
```bash
python app.py                  # opens http://127.0.0.1:7860
FER_SHARE=1 python app.py      # creates a temporary public share link
```

**Visualize the dataset:**
```bash
python plot_distribution.py
python plot_distribution.py --by-usage
```

## Tech stack

TensorFlow / Keras · OpenCV · scikit-learn · pandas · matplotlib / seaborn ·
Gradio · (optional PyTorch pipeline)

## License

MIT
