# constants, loading, assumption checks, submission write

from __future__ import annotations

from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
ID_COL, BATCH_COL, CONC_COL, TARGET_COL = "measurement_id", "batch", "concentration", "gas_class"
N_SENSORS, N_DESCRIPTORS = 16, 8
FEATURE_COLS = [f"feat_{i}" for i in range(1, N_SENSORS * N_DESCRIPTORS + 1)]   


TRAIN_BATCHES = tuple(range(1, 10))
TEST_BATCH = 10

CLASS_NAMES ={
    1: "Ethanol",
    2: "Ethylene",
    3: "Ammonia",
    4: "Acetaldehyde",
    5: "Acetone",
    6: "Toluene",
}

def check_assumptions(train, test):
    assert list(train[FEATURE_COLS].shape)[1] == 128
    assert not train[FEATURE_COLS].isna().any().any()
    assert set(train[BATCH_COL].unique()) == set(TRAIN_BATCHES)
    assert set(test[BATCH_COL].unique()) == {TEST_BATCH}
    assert set(train[TARGET_COL].unique()) == set(CLASS_NAMES)     # {1..6} overall

    # NOT asserted — true but not a hard requirement, so just surfaced:
    counts = train[TARGET_COL].value_counts()
    print("class counts (imbalanced — this is why macro-F1, not accuracy):\n", counts)
    for b in TRAIN_BATCHES:
        missing = set(CLASS_NAMES) - set(train.loc[train[BATCH_COL] == b, TARGET_COL])
        if missing:
            print(f"batch {b} has no rows for class(es): {missing}")

def write_submission(pred_df, out_dir="output"):
    Path(out_dir).mkdir(exist_ok=True)
    existing = sorted(Path(out_dir).glob("submission_*.csv"))
    n = int(existing[-1].stem.split("_")[1]) + 1 if existing else 1
    path = Path(out_dir) / f"submission_{n:02d}.csv"
    pred_df.to_csv(path, index=False)