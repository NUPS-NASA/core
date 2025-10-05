#!/usr/bin/env python3
"""Evaluate or run inference with a trained HybridScatterClassifier."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import List

from nups_core.classify import (
    HybridScatterClassifier,
    load_labeled_scatter_directory,
    load_scatter_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate or apply the scatter classifier")
    parser.add_argument("model", type=Path, help="Path to the trained model pickle")
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Directory with CSV files. If it contains 'positive'/'negative' subfolders, metrics are computed.",
    )
    parser.add_argument("--csv", type=Path, help="Classify a single CSV file")
    parser.add_argument(
        "--show-details",
        action="store_true",
        help="Include intermediate details (stage1 label, score, ML probability) in the output",
    )

    args = parser.parse_args()
    if not args.csv and not args.data_dir:
        parser.error("Provide either --csv or --data-dir for evaluation")
    return args


def _load_model(path: Path) -> HybridScatterClassifier:
    with path.open("rb") as handle:
        model = pickle.load(handle)
    if not isinstance(model, HybridScatterClassifier):
        raise TypeError(f"Pickle at {path} is not a HybridScatterClassifier instance")
    return model


def _format_prediction(pred: dict, show_details: bool) -> str:
    parts = [f"label={pred['label']}"]
    if show_details:
        parts.append(f"stage1={pred['stage1_label']}")
        parts.append(f"score={pred['score']:.3f}")
        if "ml_proba" in pred:
            parts.append(f"ml_proba={pred['ml_proba']:.3f}")
    return ", ".join(parts)


def _predict_single(model: HybridScatterClassifier, csv_path: Path, show_details: bool) -> None:
    x_vals, y_vals = load_scatter_csv(csv_path)
    prediction = model.predict(x_vals, y_vals)
    formatted = _format_prediction(prediction, show_details)
    print(f"{csv_path}: {formatted}")


def _evaluate_labeled_directory(
    model: HybridScatterClassifier,
    directory: Path,
    show_details: bool,
) -> None:
    samples, labels, paths = load_labeled_scatter_directory(directory)
    predictions = [model.predict(x_vals, y_vals) for x_vals, y_vals in samples]

    predicted_labels: List[int] = [1 if pred["label"] == "positive" else 0 for pred in predictions]
    tp = sum(1 for y_hat, y_true in zip(predicted_labels, labels) if y_hat == 1 and y_true == 1)
    tn = sum(1 for y_hat, y_true in zip(predicted_labels, labels) if y_hat == 0 and y_true == 0)
    fp = sum(1 for y_hat, y_true in zip(predicted_labels, labels) if y_hat == 1 and y_true == 0)
    fn = sum(1 for y_hat, y_true in zip(predicted_labels, labels) if y_hat == 0 and y_true == 1)
    correct = tp + tn
    ambiguous = sum(1 for pred in predictions if pred["stage1_label"] == "ambiguous")

    total = len(labels)
    print(f"Evaluated {total} labeled samples from {directory}")
    print(f"Accuracy: {correct / total:.3f}")
    print(f"Stage1 ambiguous rate: {ambiguous / total:.3f}")
    print(f"TP={tp}, TN={tn}, FP={fp}, FN={fn}")

    if show_details:
        for path, pred, true_label in zip(paths, predictions, labels):
            formatted = _format_prediction(pred, show_details=True)
            label_name = "positive" if true_label == 1 else "negative"
            print(f"{path}: true={label_name}, {formatted}")


def _predict_directory(model: HybridScatterClassifier, directory: Path, show_details: bool) -> None:
    csv_files = sorted(p for p in directory.rglob("*.csv") if p.is_file())
    if not csv_files:
        raise ValueError(f"No CSV files found in {directory}")

    for csv_path in csv_files:
        try:
            _predict_single(model, csv_path, show_details)
        except Exception as exc:  # pragma: no cover - surfaced to the user for visibility
            print(f"[warn] Skipping {csv_path}: {exc}", file=sys.stderr)


def main() -> None:
    args = parse_args()
    model_path = args.model.resolve()
    clf = _load_model(model_path)

    if args.csv:
        _predict_single(clf, args.csv.resolve(), args.show_details)

    if args.data_dir:
        data_dir = args.data_dir.resolve()
        pos_dir = data_dir / "positive"
        neg_dir = data_dir / "negative"
        if pos_dir.is_dir() and neg_dir.is_dir():
            _evaluate_labeled_directory(clf, data_dir, args.show_details)
        else:
            _predict_directory(clf, data_dir, args.show_details)


if __name__ == "__main__":
    main()
