"""
Visualize the FER2013 class distribution as a bar chart.

Reads fer2013.csv, counts images per emotion, and saves a labeled bar chart.
With --by-usage it also shows the Training / PublicTest / PrivateTest split as
grouped bars.

Usage:
    python plot_distribution.py
    python plot_distribution.py --by-usage
    python plot_distribution.py --csv fer2013.csv --out fer2013_distribution.png
"""

import argparse

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

EMOTION_NAMES = {0: "anger", 1: "disgust", 2: "fear", 3: "happiness",
                 4: "sadness", 5: "surprise", 6: "neutral"}


def main():
    ap = argparse.ArgumentParser(description="Bar chart of FER2013 class distribution")
    ap.add_argument("--csv", default="fer2013.csv")
    ap.add_argument("--out", default="fer2013_distribution.png")
    ap.add_argument("--by-usage", action="store_true",
                    help="Group bars by the Usage column (train/test splits)")
    args = ap.parse_args()

    # Only read the columns we need (fast — avoids loading the huge pixel strings).
    cols = ["emotion", "Usage"] if args.by_usage else ["emotion"]
    df = pd.read_csv(args.csv, usecols=cols)
    df["emotion_name"] = df["emotion"].map(EMOTION_NAMES)
    order = [EMOTION_NAMES[i] for i in range(7)]  # fixed emotion order

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))

    if args.by_usage:
        ax = sns.countplot(data=df, x="emotion_name", hue="Usage", order=order)
        title = "FER2013 class distribution by data split"
    else:
        counts = df["emotion_name"].value_counts().reindex(order)
        ax = sns.barplot(x=counts.index, y=counts.values,
                         hue=counts.index, palette="viridis", legend=False)
        # annotate each bar with its count + percentage
        total = int(counts.sum())
        for p, val in zip(ax.patches, counts.values):
            ax.annotate(f"{int(val)}\n{val / total:.1%}",
                        (p.get_x() + p.get_width() / 2, p.get_height()),
                        ha="center", va="bottom", fontsize=10)
        title = f"FER2013 class distribution  (n = {total:,})"

    ax.set_title(title, fontsize=14, weight="bold")
    ax.set_xlabel("Emotion")
    ax.set_ylabel("Number of images")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")

    # Also print the table to the console.
    print("\nCounts per emotion:")
    print(df["emotion_name"].value_counts().reindex(order).to_string())


if __name__ == "__main__":
    main()
