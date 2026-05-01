import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from data_utils import (
    DEFAULT_LABEL_COLUMN,
    DEFAULT_TEXT_COLUMN,
    load_splits,
    save_json,
    save_predictions,
)


class TweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts.astype(str).tolist()
        self.labels = labels.astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        encoded = self.tokenizer(
            self.texts[index],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        return item


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune BERTweet for tweet-level depression detection."
    )

    parser.add_argument(
        "--split-dir",
        default="data/processed",
        help="Directory containing train/dev/test CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/bertweet",
        help="Directory where metrics, predictions, and logs will be saved.",
    )
    parser.add_argument(
        "--model-name",
        default="vinai/bertweet-base",
        help="Hugging Face model name.",
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
        "--max-length",
        type=int,
        default=128,
        help="Maximum token length.",
    )
    parser.add_argument(
        "--epochs",
        type=float,
        default=2.0,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Training/evaluation batch size.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--max-train-examples",
        type=int,
        default=None,
        help="Optional cap for quick debugging. Example: 500.",
    )

    return parser.parse_args()


def make_debug_subset(train_df, label_column, max_train_examples):
    if max_train_examples is None:
        return train_df.reset_index(drop=True)

    sample_size = min(max_train_examples, len(train_df))

    if sample_size >= len(train_df):
        return train_df.reset_index(drop=True)

    _, sampled_df = train_test_split(
        train_df,
        test_size=sample_size,
        random_state=42,
        stratify=train_df[label_column],
    )

    return sampled_df.reset_index(drop=True)


def compute_transformer_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        zero_division=0,
    )

    accuracy = accuracy_score(labels, predictions)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


def evaluate_and_save_predictions(
    trainer,
    split_name,
    split_df,
    dataset,
    text_column,
    label_column,
    output_dir,
):
    prediction_output = trainer.predict(dataset)
    logits = prediction_output.predictions
    predictions = np.argmax(logits, axis=1)
    labels = split_df[label_column].astype(int).to_numpy()

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        zero_division=0,
    )

    metrics = {
        "accuracy": round(float(accuracy_score(labels, predictions)), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
        "classification_report": classification_report(
            labels,
            predictions,
            zero_division=0,
            output_dict=True,
        ),
        "split": split_name,
        "num_examples": int(len(split_df)),
    }

    save_json(metrics, output_dir / f"{split_name}_metrics.json")
    save_predictions(
        split_df,
        predictions,
        output_dir / f"{split_name}_predictions.csv",
        text_column=text_column,
        label_column=label_column,
    )

    return metrics


def build_dataset(split_df, tokenizer, text_column, label_column, max_length):
    return TweetDataset(
        split_df[text_column],
        split_df[label_column],
        tokenizer,
        max_length,
    )


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = load_splits(args.split_dir)

    train_df = make_debug_subset(
        splits["train"],
        args.label_column,
        args.max_train_examples,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=False,
        normalization=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
    )

    train_dataset = build_dataset(
        train_df,
        tokenizer,
        args.text_column,
        args.label_column,
        args.max_length,
    )

    dev_dataset = build_dataset(
        splits["dev"],
        tokenizer,
        args.text_column,
        args.label_column,
        args.max_length,
    )

    test_dataset = build_dataset(
        splits["test"],
        tokenizer,
        args.text_column,
        args.label_column,
        args.max_length,
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",
        logging_dir=str(output_dir / "logs"),
        logging_steps=50,
        report_to="none",
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_transformer_metrics,
    )

    trainer.train()

    results = {
        "train": evaluate_and_save_predictions(
            trainer,
            "train",
            train_df,
            train_dataset,
            args.text_column,
            args.label_column,
            output_dir,
        ),
        "dev": evaluate_and_save_predictions(
            trainer,
            "dev",
            splits["dev"],
            dev_dataset,
            args.text_column,
            args.label_column,
            output_dir,
        ),
        "test": evaluate_and_save_predictions(
            trainer,
            "test",
            splits["test"],
            test_dataset,
            args.text_column,
            args.label_column,
            output_dir,
        ),
    }

    summary = {
        "model": "bertweet",
        "model_name": args.model_name,
        "split_dir": str(args.split_dir),
        "output_dir": str(output_dir),
        "hyperparameters": {
            "max_length": args.max_length,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "max_train_examples": args.max_train_examples,
        },
        "train_examples_used": int(len(train_df)),
        "results": results,
    }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()