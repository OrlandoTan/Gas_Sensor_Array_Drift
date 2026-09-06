
from __future__ import annotations

import numpy as np
import pandas as pd

from data import FEATURE_COLS, N_DESCRIPTORS, N_SENSORS


def as_sensor_cube(df: pd.DataFrame) -> np.ndarray:
    """(n_rows, 16, 8) view of the feature block."""
    return df[FEATURE_COLS].to_numpy(float).reshape(len(df), N_SENSORS, N_DESCRIPTORS)


def features(df: pd.DataFrame, state_fill: np.ndarray | None = None):
    """Return (X, state_fill).

    state_fill is the per-sensor median of logS learned on the source frame. since
    logS isnt stacked into X (see the module docstring) it doesnt currently change
    the returned features at all, the plumbing is just kept around so the block can
    be switched back on later without touching the calling code in train.py.
    """
    cube = as_sensor_cube(df)
    dr, ratio = cube[:, :, 0], cube[:, :, 1]

    # L1 over the 16 sensors, per row. this is the part that actually does the
    # heavy lifting against drift, because it doesnt care what the absolute scale
    # of a sensor is, only how it compares to the other 15 in that same row.
    l1 = np.abs(cube).sum(axis=1, keepdims=True) + 1e-6
    pattern = (cube / l1).reshape(len(df), -1)

    logscale = np.log1p(np.abs(dr)).mean(axis=1, keepdims=True)
    logconc = np.log(df["concentration"].to_numpy(float))[:, None]

    # logS is supposed to estimate R0, the sensors baseline resistance, from
    # S = dR / (ratio - 1). found out the hard way that ratio hits exactly 1.0 for
    # a chunk of rows, which is a divide by zero, and a bunch more rows sit close
    # enough to 1 that S blows up to something huge and useless. the conditions
    # below (dr > 0, ratio > 1.02, S > 0) are just guardrails to throw out the
    # garbage values before they poison the median fill.
    with np.errstate(divide="ignore", invalid="ignore"):
        S = dr / (ratio - 1.0)  # = R0, the baseline resistance
        logS = np.log(np.where((dr > 0) & (ratio > 1.02) & (S > 0), S, np.nan))

    if state_fill is None:
        state_fill = np.nanmedian(logS, axis=0)
    logS = np.where(np.isfinite(logS), logS, state_fill)

    X = np.hstack(
        [pattern, logscale, logconc]  # tested with logS included, no improvement, left out on purpose
    )
    return X, state_fill
