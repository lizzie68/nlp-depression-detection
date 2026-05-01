import argparse
import json
from pathlib import Path

from data_utils import (
    DEFAULT_LABEL_COLUMN,
    DEFAULT_TEXT_COLUMN,
    compute_metrics,
    load_splits,
    save_json,
    save_predictions,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate a majority-class baseline."
    )
    parser.add_argument(
        "--split-dir",
        default="data/processed",
        help="Directory containing train/dev/test CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/majority",
        help="Directory where metrics and predictions will be saved.",
    )
    parser.add_argument(
        "--label-column",
        default=DEFAULT_LABEL_COLUMN,
        help="Name of the label column in the split files.",
    )
    parser.add_argument(
        "--text-column",
        default=DEFAULT_TEXT_COLUMN,
        help="Name of the text column in the split files.",
    )
    return parser.parse_args()


def majority_label(train_df, label_column):
    return int(train_df[label_column].mode().iloc[0])


def evaluate_split(split_name, split_df, label, text_column, label_column, output_dir):
    predictions = [label] * len(split_df)
    metrics = compute_metrics(split_df[label_column], predictions)
    metrics["split"] = split_name
    metrics["majority_label"] = label
    metrics["num_examples"] = int(len(split_df))

    save_json(metrics, output_dir / f"{split_name}_metrics.json")
    save_predictions(
        split_df,
        predictions,
        output_dir / f"{split_name}_predictions.csv",
        text_column=text_column,
        label_column=label_column,
    )
    return metrics


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    splits = load_splits(args.split_dir)
    label = majority_label(splits["train"], args.label_column)

    results = {
        split_name: evaluate_split(
            split_name,
            split_df,
            label,
            args.text_column,
            args.label_column,
            output_dir,
        )
        for split_name, split_df in splits.items()
    }

    summary = {
        "model": "majority_baseline",
        "split_dir": str(args.split_dir),
        "output_dir": str(output_dir),
        "majority_label": label,
        "results": results,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
