import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


DEFAULT_TEXT_COLUMN = "post_text"
DEFAULT_LABEL_COLUMN = "label"


def load_and_prepare_dataset(
    data_path,
    text_column=DEFAULT_TEXT_COLUMN,
    label_column=DEFAULT_LABEL_COLUMN,
):
    data_path = Path(data_path)
    df = pd.read_csv(data_path)
    dataset = df[[text_column, label_column]].dropna().copy()
    dataset[text_column] = dataset[text_column].astype(str).str.strip()
    dataset = dataset[dataset[text_column] != ""].reset_index(drop=True)
    dataset[label_column] = dataset[label_column].astype(int)
    dataset.insert(0, "example_id", dataset.index)
    return dataset


def create_splits(
    dataset,
    label_column=DEFAULT_LABEL_COLUMN,
    train_size=0.8,
    dev_size=0.1,
    test_size=0.1,
    random_state=42,
):
    total = train_size + dev_size + test_size
    if abs(total - 1.0) > 1e-9:
        raise ValueError(
            f"Split sizes must sum to 1.0, got {train_size} + {dev_size} + {test_size} = {total}"
        )

    train_df, temp_df = train_test_split(
        dataset,
        test_size=(1.0 - train_size),
        random_state=random_state,
        stratify=dataset[label_column],
    )

    relative_test_size = test_size / (dev_size + test_size)
    dev_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_size,
        random_state=random_state,
        stratify=temp_df[label_column],
    )

    return {
        "train": train_df.reset_index(drop=True),
        "dev": dev_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }


def save_splits(splits, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = {}
    for split_name, split_df in splits.items():
        path = output_dir / f"{split_name}.csv"
        split_df.to_csv(path, index=False)
        saved_paths[split_name] = str(path)
    return saved_paths


def load_splits(split_dir):
    split_dir = Path(split_dir)
    return {
        "train": pd.read_csv(split_dir / "train.csv"),
        "dev": pd.read_csv(split_dir / "dev.csv"),
        "test": pd.read_csv(split_dir / "test.csv"),
    }


def compute_metrics(y_true, y_pred):
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            zero_division=0,
            output_dict=True,
        ),
    }


def save_json(payload, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))


def save_predictions(
    split_df,
    predictions,
    output_path,
    text_column=DEFAULT_TEXT_COLUMN,
    label_column=DEFAULT_LABEL_COLUMN,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_df = split_df[["example_id", text_column, label_column]].copy()
    predictions_df["prediction"] = predictions
    predictions_df.to_csv(output_path, index=False)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create and save train/dev/test splits for the mental health dataset."
    )
    parser.add_argument(
        "--data",
        default="Mental-Health-Twitter.csv",
        help="Path to the raw CSV dataset.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Directory where train/dev/test CSV files will be saved.",
    )
    parser.add_argument(
        "--text-column",
        default=DEFAULT_TEXT_COLUMN,
        help="Text column to keep in the processed dataset.",
    )
    parser.add_argument(
        "--label-column",
        default=DEFAULT_LABEL_COLUMN,
        help="Label column to keep in the processed dataset.",
    )
    parser.add_argument(
        "--train-size",
        type=float,
        default=0.8,
        help="Fraction of examples in the training split.",
    )
    parser.add_argument(
        "--dev-size",
        type=float,
        default=0.1,
        help="Fraction of examples in the development split.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.1,
        help="Fraction of examples in the test split.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible splitting.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = load_and_prepare_dataset(
        data_path=args.data,
        text_column=args.text_column,
        label_column=args.label_column,
    )
    splits = create_splits(
        dataset,
        label_column=args.label_column,
        train_size=args.train_size,
        dev_size=args.dev_size,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    saved_paths = save_splits(splits, args.output_dir)
    summary = {
        "rows_used": int(len(dataset)),
        "split_sizes": {name: int(len(split_df)) for name, split_df in splits.items()},
        "saved_paths": saved_paths,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
