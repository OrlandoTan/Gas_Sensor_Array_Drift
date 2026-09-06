"""Batch-aware validation, the only kind that actually predicts the real score here.

a normal random train/test split mixes measurements from the same batch into both
sides, so a model that just memorizes batch-specific offsets scores great on the
split and then falls apart on batch 10 for real. both schemes below hold out whole
batches so that trap cant happen.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from data import CLASS_NAMES, CLASSES


def macro_f1(y_true, y_pred) -> float:
    """The competition metric. unweighted mean of the six per-class F1 scores."""
    return float(f1_score(y_true, y_pred, average="macro", labels=CLASSES, zero_division=0))


def leave_one_batch_out(batch: np.ndarray):
    """Hold out each batch in turn, train on all the others.

    just a diagnostic
    """
    for b in np.unique(batch):
        val = batch == b
        yield b, np.flatnonzero(~val), np.flatnonzero(val)


def forward_chaining(batch: np.ndarray, min_train_batches: int = 3):
    """Train on batches <= k, validate on batch k+1.

    this is the honest one. every fold here only ever predicts a LATER batch from
    EARLIER ones, which is exactly the situation batch 10 is actually in (trained on
    all of 1-9, nothing after it to lean on). if you only have time to trust one
    validation scheme, trust this one over leave_one_batch_out.
    """
    batches = np.unique(batch)
    for i in range(min_train_batches, len(batches)):
        yield batches[i], np.flatnonzero(np.isin(batch, batches[:i])), np.flatnonzero(batch == batches[i])


SCHEMES = {"forward": forward_chaining, "lobo": leave_one_batch_out}


def evaluate(model, X: pd.DataFrame, y: np.ndarray, batch: np.ndarray, scheme: str = "forward") -> pd.DataFrame:
    """Run a batch-aware CV and return per-fold macro F1.

    the mean here is unweighted across folds, so a tiny batch counts just as much as
    a big one. pay the most attention to the last folds, theyre the closest thing we
    have to a rehearsal of batch 10. also remember batches 3-5 have zero Toluene rows,
    so a fold touching those can only ever be as good as the classes it actually has.
    """
    if scheme not in SCHEMES:
        raise ValueError(f"scheme must be one of {sorted(SCHEMES)}, got {scheme!r}")

    rows = []
    for held_out, tr, va in SCHEMES[scheme](batch):
        fold = clone(model)
        fold.fit(X.iloc[tr], y[tr])
        pred = fold.predict(X.iloc[va])
        rows.append(
            {
                "held_out_batch": held_out,
                "n_train": len(tr),
                "n_val": len(va),
                "classes_present": len(np.unique(y[va])),
                "macro_f1": macro_f1(y[va], pred),
            }
        )
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> str:
    mean = results["macro_f1"].mean()
    last = results["macro_f1"].iloc[-1]
    table = results.to_string(index=False, float_format="%.4f")
    return f"{table}\n\nmean macro-F1 {mean:.4f} | last fold {last:.4f}"


def per_class_report(y_true, y_pred) -> str:
    return classification_report(
        y_true,
        y_pred,
        labels=CLASSES,
        target_names=[CLASS_NAMES[c] for c in CLASSES],
        zero_division=0,
    )


def confusion(y_true, y_pred) -> pd.DataFrame:
    names = [CLASS_NAMES[c] for c in CLASSES]
    return pd.DataFrame(confusion_matrix(y_true, y_pred, labels=CLASSES), index=names, columns=names)
