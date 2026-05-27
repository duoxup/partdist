"""Transformations on SliceDistribution.

Centroid shifts, Twiss matching, linear correlations (dispersion,
rotation), and RMS scaling on a single transverse slice.

Sibling of partdist.pd3d.manipulator, restricted to operations meaningful
on a zero-thickness slice (no chirp / chicane / longitudinal replication).

Default `weight="lam_abs"` for every operation that accepts a weight —
charge-weighted second moments are the physical convention for Twiss and
related diagnostics on non-uniform-lam beams.

All public functions follow the (dist, *, inplace=False, weight=...,
mask=None, ...) shape and return a SliceDistribution (a new instance
unless inplace=True).
"""
from __future__ import annotations

from typing import Literal, Optional, Sequence, Union

import numpy as np

from ..pd3d.utils import (
    _copy_or_inplace,
    _normalize_mask,
    _get_weight_array,
    _extract_data,
)
from ..pd3d.manipulator import (
    _weighted_raw_twiss_from_arrays,
    _weighted_centered_twiss_from_arrays,
)
from .core import SliceDistribution


ArrayLike = Union[np.ndarray, Sequence[float], float]
