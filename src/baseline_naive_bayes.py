import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


TEXT_CANDIDATES = [
    "tweet",
    "tweets",
    "text",
    "post",
    "sentence",
    "content",
]

LABEL_CANDIDATES = [
    "label",
    "target",
    "class",
    "depression",
    "status",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate a Naive Bayes baseline for depression detection."
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to the CSV dataset downloaded from Kaggle.",
    )
    parser.add_argument(
        "--text-column",
        default=None,
        help="Name of the text column. Auto-detected when omitted.",
    )
    parser.add_argument(
        "--label-column",
        default=None,
        help="Name of the label column. Auto-detected when omitted.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of rows used for evaluation.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used for the train/test split.",
    )
    parser.add_argument(
        "--min-df",
        type=int,
        default=1,
        help="Minimum document frequency passed to TfidfVectorizer.",
    )
    return parser.parse_args()


def find_column(columns, candidates, explicit_name, column_kind):
    if explicit_name:
        if explicit_name not in columns:
            raise ValueError(
                f"{column_kind} column '{explicit_name}' was not found. "
                f"Available columns: {list(columns)}"
            )
        return explicit_name

    lowered = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]

    raise ValueError(
        f"Could not infer the {column_kind} column. "
        f"Available columns: {list(columns)}"
    )


def normalize_binary_labels(series):
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        unique_values = sorted(numeric.astype(int).unique().tolist())
        if len(unique_values) != 2:
            raise ValueError(
                f"Expected exactly 2 label values, found {unique_values}."
            )
        mapping = {unique_values[0]: 0, unique_values[1]: 1}
        normalized = numeric.astype(int).map(mapping)
        label_names = {0: str(unique_values[0]), 1: str(unique_values[1])}
        return normalized.astype(int), label_names

    cleaned = series.astype(str).str.strip().str.lower()
    unique_values = sorted(cleaned.unique().tolist())
    if len(unique_values) != 2:
        raise ValueError(f"Expected exactly 2 label values, found {unique_values}.")

    positive_tokens = {
        "1",
        "true",
        "yes",
        "depressed",
        "depression",
        "positive",
    }
    negative_tokens = {
        "0",
        "false",
        "no",
        "not depressed",
        "non-depressed",
        "negative",
        "normal",
    }

    positive_value = next((value for value in unique_values if value in positive_tokens), None)
    negative_value = next((value for value in unique_values if value in negative_tokens), None)

    if positive_value is None or negative_value is None:
        negative_value, positive_value = unique_values

    mapping = {negative_value: 0, positive_value: 1}
    normalized = cleaned.map(mapping)
    if normalized.isna().any():
        raise ValueError(
            "Unable to normalize labels into a binary 0/1 target. "
            f"Observed values: {unique_values}"
        )

    label_names = {0: negative_value, 1: positive_value}
    return normalized.astype(int), label_names


def main():
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    if df.empty:
        raise ValueError("Dataset is empty.")

    text_column = find_column(
        df.columns, TEXT_CANDIDATES, args.text_column, "text"
    )
    label_column = find_column(
        df.columns, LABEL_CANDIDATES, args.label_column, "label"
    )

    dataset = df[[text_column, label_column]].dropna().copy()
    dataset[text_column] = dataset[text_column].astype(str).str.strip()
    dataset = dataset[dataset[text_column] != ""]
    labels, label_names = normalize_binary_labels(dataset[label_column])

    X_train, X_test, y_train, y_test = train_test_split(
        dataset[text_column],
        labels,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=labels,
    )

    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=args.min_df,
                ),
            ),
            ("classifier", MultinomialNB()),
        ]
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "dataset_path": str(data_path),
        "rows_used": int(len(dataset)),
        "text_column": text_column,
        "label_column": label_column,
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "accuracy": round(accuracy_score(y_test, predictions), 4),
        "precision": round(precision_score(y_test, predictions, zero_division=0), 4),
        "recall": round(recall_score(y_test, predictions, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, predictions, zero_division=0), 4),
        "label_mapping": label_names,
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            target_names=[label_names[0], label_names[1]],
            zero_division=0,
            output_dict=True,
        ),
    }

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
