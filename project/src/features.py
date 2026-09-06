"""
1. 128 feature pattern (which sensors responded relative to each other (the gas fingerprint, scale free, L1 normalized over the 
    16 sensors, per row, so it survives drift without needing to know the true concentration or the sensors baseline))
2. Ranks 128 which is the same 16-sensor comparison as pattern, but by rank order instead of magnitude 
3. logscale     
4. logconc      
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data import FEATURE_COLS, N_DESCRIPTORS, N_SENSORS


def as_sensor_cube(df: pd.DataFrame) -> np.ndarray:
    """(n_rows, 16, 8) view of the feature block."""
    return df[FEATURE_COLS].to_numpy(float).reshape(len(df), N_SENSORS, N_DESCRIPTORS)


def features(df: pd.DataFrame, state_fill: np.ndarray | None = None):
    cube = as_sensor_cube(df)
    dr, ratio = cube[:, :, 0], cube[:, :, 1]

    # L1 doesnt care what the absolute scale of a sensor is, only how it 
    # compares to the other 15 in that same row.
    l1 = np.abs(cube).sum(axis=1, keepdims=True) + 1e-6
    pattern = (cube / l1).reshape(len(df), -1)

    # rank block: for each descriptor, replace every sensor's value by its rank
    # (0..15) among the 16 sensors in that row. any drift that squashes or
    # stretches a descriptor's values but keeps their order leaves these columns
    # untouched, so it survives batch 10's post-shock geometry better than pattern
    # alone. kept ALONGSIDE pattern, not instead of it. replaces the earlier dynscale approrach
    ranks = cube.argsort(axis=1).argsort(axis=1).reshape(len(df), -1) / (N_SENSORS - 1)

    logscale = np.log1p(np.abs(dr)).mean(axis=1, keepdims=True)
    logconc = np.log(df["concentration"].to_numpy(float))[:, None]

    with np.errstate(divide="ignore", invalid="ignore"):
        S = dr / (ratio - 1.0)
        logS = np.log(np.where((dr > 0) & (ratio > 1.02) & (S > 0), S, np.nan))

    if state_fill is None:
        state_fill = np.nanmedian(logS, axis=0)
    logS = np.where(np.isfinite(logS), logS, state_fill)

    X = np.hstack(
        [pattern, ranks, logscale, logconc]  # tested with logS included, no improvement
    )
    return X, state_fill
