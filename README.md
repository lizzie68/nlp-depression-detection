# nlp-depression-detection

Naive Bayes baseline for tweet-level depression detection using the Kaggle dataset:
https://www.kaggle.com/datasets/infamouscoder/mental-health-social-media/data

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the baseline

Download the CSV from Kaggle, then point the script at it:

```bash
python3 baseline_naive_bayes.py --data /path/to/your_dataset.csv
```

If the dataset uses different column names than the defaults, pass them explicitly:

```bash
python3 baseline_naive_bayes.py \
  --data /path/to/your_dataset.csv \
  --text-column tweet \
  --label-column label
```

The script prints:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Full classification report
