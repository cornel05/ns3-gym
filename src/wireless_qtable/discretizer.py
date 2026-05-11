from __future__ import annotations

from typing import Iterable

import numpy as np


def discretize_observation(
    observation: Iterable[float],
    bins: int = 10,
    low: float = 0.0,
    high: float = 1.0,
) -> tuple[int, ...]:
    """Convert observation to a discrete tuple key for the Q-table.

    For common cognitive-radio occupancy vectors (0/1), this returns tuple(int).
    For continuous values, this applies simple uniform binning.
    """
    arr = np.asarray(observation)

    # Already discrete or binary.
    if np.issubdtype(arr.dtype, np.integer) or np.all(np.isin(arr, [0, 1])):
        return tuple(int(x) for x in arr.tolist())

    # Fallback for continuous observations.
    clipped = np.clip(arr.astype(float), low, high)
    scaled = (clipped - low) / max(high - low, 1e-12)
    idx = np.floor(scaled * bins).astype(int)
    idx = np.clip(idx, 0, bins - 1)
    return tuple(int(x) for x in idx.tolist())
