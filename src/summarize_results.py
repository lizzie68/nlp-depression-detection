import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_MODELS = {
    "majority": "outputs/majority",
    "naive_bayes": "outputs/naive_bayes",
    "logistic_regression": "outputs/logistic_regression",
    "bertweet": "outputs/bertweet",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize train/dev/test metrics from all models into one CSV."
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Base output directory containing model result folders.",
    )
    parser.add_argument(
        "--summary-path",
        default="outputs/metrics_summary.csv",
        help="Path where the combined metrics CSV will be saved.",
    )
    return parser.parse_args()


def load_metric_file(path):
    with open(path, "r") as f:
        return json.load(f)


def collect_metrics(output_dir):
    output_dir = Path(output_dir)
    rows = []

    for model_name, model_dir in DEFAULT_MODELS.items():
        model_path = Path(model_dir)

        if not model_path.exists():
            print(f"Skipping {model_name}: {model_path} does not exist.")
            continue

        for split in ["train", "dev", "test"]:
            metric_path = model_path / f"{split}_metrics.json"

            if not metric_path.exists():
                print(f"Skipping missing file: {metric_path}")
                continue

            metrics = load_metric_file(metric_path)

            rows.append(
                {
                    "model": model_name,
                    "split": split,
                    "accuracy": metrics.get("accuracy"),
                    "precision": metrics.get("precision"),
                    "recall": metrics.get("recall"),
                    "f1_score": metrics.get("f1_score"),
                    "num_examples": metrics.get("num_examples"),
                    "confusion_matrix": metrics.get("confusion_matrix"),
                }
            )

    return pd.DataFrame(rows)


def main():
    args = parse_args()

    summary_df = collect_metrics(args.output_dir)

    if summary_df.empty:
        raise ValueError("No metrics found. Make sure model scripts have been run.")

    split_order = {"train": 0, "dev": 1, "test": 2}
    summary_df["split_order"] = summary_df["split"].map(split_order)
    summary_df = summary_df.sort_values(["split_order", "model"]).drop(
        columns=["split_order"]
    )

    summary_path = Path(args.summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)

    print(summary_df.to_string(index=False))
    print(f"\nSaved metrics summary to: {summary_path}")


if __name__ == "__main__":
    main()