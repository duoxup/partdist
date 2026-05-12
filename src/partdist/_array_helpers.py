"""Internal helpers for converting user input to 1D numpy arrays.

Single source of truth — do not redefine in submodules.
"""
from __future__ import annotations

from typing import Iterable, Sequence, Union

import numpy as np


ArrayLike = Union[float, Sequence[float], Iterable[float], np.ndarray]


def as_1d_array(
    data: ArrayLike,
    *,
    dtype: type | None = None,
    name: str = "array",
) -> np.ndarray:
    """Cast ``data`` to a 1D numpy array with optional dtype coercion.

    Scalars (ndim == 0) are reshaped to length-1 arrays.
    Raises ``ValueError`` if the resulting array is not exactly 1-dimensional.
    """
    arr = np.asarray(data, dtype=dtype)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim != 1:
        raise ValueError(
            f"{name} must be a 1D array, got ndim={arr.ndim} (shape={arr.shape})."
        )
    return arr


def as_1d_float_array(data: ArrayLike, name: str = "array") -> np.ndarray:
    """Shortcut for ``as_1d_array(data, dtype=float, name=name)``."""
    return as_1d_array(data, dtype=float, name=name)
