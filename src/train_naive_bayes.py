import argparse
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
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
        description="Train and evaluate a Naive Bayes baseline on precomputed splits."
    )
    parser.add_argument(
        "--split-dir",
        default="data/processed",
        help="Directory containing train/dev/test CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/naive_bayes",
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
        "--min-df",
        type=int,
        default=1,
        help="Minimum document frequency passed to TfidfVectorizer.",
    )
    return parser.parse_args()


def build_model(min_df):
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=min_df,
                ),
            ),
            ("classifier", MultinomialNB()),
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

    model = build_model(args.min_df)
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
        "model": "naive_bayes",
        "split_dir": str(args.split_dir),
        "output_dir": str(output_dir),
        "results": results,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
