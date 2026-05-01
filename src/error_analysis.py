import argparse
from pathlib import Path

import pandas as pd


MODEL_PREDICTION_PATHS = {
    "majority": "outputs/majority/test_predictions.csv",
    "naive_bayes": "outputs/naive_bayes/test_predictions.csv",
    "logistic_regression": "outputs/logistic_regression/test_predictions.csv",
    "bertweet": "outputs/bertweet/test_predictions.csv",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate error analysis examples from model prediction files."
    )
    parser.add_argument(
        "--output-path",
        default="outputs/error_analysis_examples.csv",
        help="Path where error analysis examples will be saved.",
    )
    parser.add_argument(
        "--summary-path",
        default="outputs/error_analysis_summary.txt",
        help="Path where text summary will be saved.",
    )
    parser.add_argument(
        "--max-examples-per-category",
        type=int,
        default=25,
        help="Maximum number of examples to save per error category.",
    )
    return parser.parse_args()


def load_prediction_file(model_name, path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction file for {model_name}: {path}")

    df = pd.read_csv(path)

    required_columns = {"example_id", "post_text", "label", "prediction"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")

    return df[["example_id", "post_text", "label", "prediction"]].rename(
        columns={"prediction": f"{model_name}_prediction"}
    )


def merge_predictions():
    merged = None

    for model_name, path in MODEL_PREDICTION_PATHS.items():
        df = load_prediction_file(model_name, path)

        if merged is None:
            merged = df
        else:
            merged = merged.merge(
                df[["example_id", f"{model_name}_prediction"]],
                on="example_id",
                how="inner",
            )

    return merged


def add_correctness_columns(df):
    for model_name in MODEL_PREDICTION_PATHS:
        df[f"{model_name}_correct"] = (
            df[f"{model_name}_prediction"] == df["label"]
        )

    return df


def collect_category(df, category_name, condition, max_examples):
    examples = df[condition].copy()
    examples.insert(0, "error_category", category_name)
    return examples.head(max_examples)


def main():
    args = parse_args()

    df = merge_predictions()
    df = add_correctness_columns(df)

    categories = []

    categories.append(
        collect_category(
            df,
            "all_models_wrong",
            (~df["naive_bayes_correct"])
            & (~df["logistic_regression_correct"])
            & (~df["bertweet_correct"]),
            args.max_examples_per_category,
        )
    )

    categories.append(
        collect_category(
            df,
            "naive_bayes_wrong_bertweet_correct",
            (~df["naive_bayes_correct"]) & (df["bertweet_correct"]),
            args.max_examples_per_category,
        )
    )

    categories.append(
        collect_category(
            df,
            "logistic_regression_wrong_bertweet_correct",
            (~df["logistic_regression_correct"]) & (df["bertweet_correct"]),
            args.max_examples_per_category,
        )
    )

    categories.append(
        collect_category(
            df,
            "bertweet_wrong_naive_bayes_correct",
            (~df["bertweet_correct"]) & (df["naive_bayes_correct"]),
            args.max_examples_per_category,
        )
    )

    categories.append(
        collect_category(
            df,
            "false_positives_bertweet",
            (df["label"] == 0) & (df["bertweet_prediction"] == 1),
            args.max_examples_per_category,
        )
    )

    categories.append(
        collect_category(
            df,
            "false_negatives_bertweet",
            (df["label"] == 1) & (df["bertweet_prediction"] == 0),
            args.max_examples_per_category,
        )
    )

    output_df = pd.concat(categories, ignore_index=True)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)

    summary_lines = []
    summary_lines.append("Error Analysis Summary")
    summary_lines.append("======================")
    summary_lines.append(f"Total merged test examples: {len(df)}")
    summary_lines.append("")

    for model_name in MODEL_PREDICTION_PATHS:
        correct = int(df[f"{model_name}_correct"].sum())
        total = int(len(df))
        summary_lines.append(
            f"{model_name}: {correct}/{total} correct ({correct / total:.4f})"
        )

    summary_lines.append("")
    summary_lines.append("Category counts:")

    category_conditions = {
        "all_models_wrong": (~df["naive_bayes_correct"])
        & (~df["logistic_regression_correct"])
        & (~df["bertweet_correct"]),
        "naive_bayes_wrong_bertweet_correct": (~df["naive_bayes_correct"])
        & (df["bertweet_correct"]),
        "logistic_regression_wrong_bertweet_correct": (
            ~df["logistic_regression_correct"]
        )
        & (df["bertweet_correct"]),
        "bertweet_wrong_naive_bayes_correct": (~df["bertweet_correct"])
        & (df["naive_bayes_correct"]),
        "false_positives_bertweet": (df["label"] == 0)
        & (df["bertweet_prediction"] == 1),
        "false_negatives_bertweet": (df["label"] == 1)
        & (df["bertweet_prediction"] == 0),
    }

    for category_name, condition in category_conditions.items():
        summary_lines.append(f"{category_name}: {int(condition.sum())}")

    summary_path = Path(args.summary_path)
    summary_path.write_text("\n".join(summary_lines))

    print("\n".join(summary_lines))
    print(f"\nSaved examples to: {output_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()