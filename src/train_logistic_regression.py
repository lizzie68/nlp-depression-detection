import argparse
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

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
        description="Train and evaluate TF-IDF + Logistic Regression."
    )
    parser.add_argument(
        "--split-dir",
        default="data/processed",
        help="Directory containing train/dev/test CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/logistic_regression",
        help="Directory where metrics and predictions will be saved.",
    )
    parser.add_argument(
        "--text-column",
        default=DEFAULT_TEXT_COLUMN,
        help="Name of the text column in the split files.",
    )
    parser.add_argument(
        "--label-column",
        default=DEFAULT_LABEL_COLUMN,
        help="Name of the label column in the split files.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=30000,
        help="Maximum number of TF-IDF features.",
    )
    parser.add_argument(
        "--min-df",
        type=int,
        default=2,
        help="Minimum document frequency for TF-IDF.",
    )
    parser.add_argument(
        "--c",
        type=float,
        default=1.0,
        help="Inverse regularization strength for Logistic Regression.",
    )
    return parser.parse_args()


def build_model(max_features, min_df, c):
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    max_features=max_features,
                    min_df=min_df,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=c,
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def evaluate_split(model, split_name, split_df, text_column, label_column, output_dir):
    predictions = model.predict(split_df[text_column])
    metrics = compute_metrics(split_df[label_column], predictions)
    metrics["split"] = split_name
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

    model = build_model(
        max_features=args.max_features,
        min_df=args.min_df,
        c=args.c,
    )

    model.fit(splits["train"][args.text_column], splits["train"][args.label_column])

    results = {
        split_name: evaluate_split(
            model,
            split_name,
            split_df,
            args.text_column,
            args.label_column,
            output_dir,
        )
        for split_name, split_df in splits.items()
    }

    summary = {
        "model": "tfidf_logistic_regression",
        "split_dir": str(args.split_dir),
        "output_dir": str(output_dir),
        "hyperparameters": {
            "max_features": args.max_features,
            "min_df": args.min_df,
            "C": args.c,
        },
        "results": results,
    }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()