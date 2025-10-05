#!/usr/bin/env python3
"""Train the HybridScatterClassifier on labeled scatter CSV files."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from nups_core.classify import HybridScatterClassifier, load_labeled_scatter_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the scatter classifier from labeled CSV files")
    parser.add_argument("data_dir", type=Path, help="Directory containing 'positive' and 'negative' subfolders")
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("scatter_classifier.pkl"),
        help="Path where the trained model pickle will be written",
    )
    parser.add_argument(
        "--train-mode",
        choices=["ambiguous_only", "all"],
        default="ambiguous_only",
        help="Sample selection strategy used during model training",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    model_path = args.model_out.resolve()

    samples, labels, _ = load_labeled_scatter_directory(data_dir)

    clf = HybridScatterClassifier()
    clf.fit(samples, labels, train_mode=args.train_mode)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as handle:
        pickle.dump(clf, handle)

    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    print(f"Training completed on {len(labels)} samples (positive={n_pos}, negative={n_neg}).")
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
