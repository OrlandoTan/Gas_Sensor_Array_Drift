# Gas Sensor Array Drift

Predicts gas class (1-6) from a 16-sensor array, across batches that drift over time.

## Required files

- `project/data/train.csv` (labelled, batches 1-9)
- `project/data/test.csv` (unlabelled, batch 10)
- `project/data/sample_submission.csv` (row order and id list for the output)
- `project/src/data.py`, `project/src/features.py`, `project/src/validate.py`, `project/src/train.py`

All four scripts must stay in the same folder since they import each other by name.

## Dependencies

- Python 3.11+
- numpy
- pandas
- scikit-learn
- scipy

No requirements.txt exists yet. Install with:

```
pip install numpy pandas scikit-learn scipy
```

## How to reproduce the final submission

Run this from `project/src`:

```
python train.py
```

This does three things, in order:

1. Loads `train.csv` and holds out batches 9 and 7 in turn to print a held-out macro-F1 score for each (sanity check, not required for the submission).
2. Builds features and trains the final model (LDA with self-training) on all of `train.csv`.
3. Predicts on `test.csv` and writes `project/output/submission_NN.csv`, where `NN` auto-increments so old runs are never overwritten. The predicted class counts are printed at the end.

The `output/` folder is not committed to git, so it will not exist until you run the script.

To pick a specific output path instead of auto-numbering:

```
python train.py --out ../output/final_submission.csv
```

## Method

- Model: `train.py` uses LDA (`solver="lsqr"`, `shrinkage=0.001`) with self-training: the model is retrained several times, each round folding in the target-batch rows it is most confident about (see `ROUNDS` in `train.py`).
- Features: built in `features.py` from the raw sensor readings. Scale-free per-row pattern features, rank-based features (robust to drift), a log-scaled magnitude summary, and log concentration.
- This is the default `--method selftrain`. A `--method baseline` (plain logistic regression, no self-training) is also available for comparison but is not what generates the final submission.

## Random seeds

No manual seed is set. Every step (StandardScaler, LDA, self-training, logistic regression) is deterministic given the same input data, so reruns produce identical results.

## Other files

- `project/notebooks/eda.ipynb`: exploratory analysis and drift figures. Not part of the prediction pipeline, useful for understanding the data.
- `project/src/validate.py`: scoring and cross-validation helpers used to sanity-check models during development.
