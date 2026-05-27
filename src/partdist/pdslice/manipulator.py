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


def shift_centroid(
    dist: SliceDistribution,
    *,
    dx: float = 0.0,
    dy: float = 0.0,
    dpx: float = 0.0,
    dpy: float = 0.0,
    dpz: float = 0.0,
    mask: Optional[Union[np.ndarray, Sequence[bool]]] = None,
    inplace: bool = False,
) -> SliceDistribution:
    """Add constant per-axis offsets to selected particles."""
    out = _copy_or_inplace(dist, inplace=inplace)
    n = len(out)
    m = _normalize_mask(mask, n)
    for key, delta in (("x", dx), ("y", dy), ("px", dpx), ("py", dpy), ("pz", dpz)):
        if delta == 0.0:
            continue
        arr = out.get_data(key).copy()
        arr[m] += delta
        out.update_quantity(key, arr)
    return out


def rotate_xy(
    dist: SliceDistribution,
    theta: float,
    *,
    mask: Optional[Union[np.ndarray, Sequence[bool]]] = None,
    inplace: bool = False,
) -> SliceDistribution:
    """Rotate the (x, y) plane by theta; (px, py) rotates consistently."""
    out = _copy_or_inplace(dist, inplace=inplace)
    n = len(out)
    m = _normalize_mask(mask, n)
    c, s = float(np.cos(theta)), float(np.sin(theta))
    for a_key, b_key in (("x", "y"), ("px", "py")):
        a = out.get_data(a_key).copy()
        b = out.get_data(b_key).copy()
        a_sel = a[m]
        b_sel = b[m]
        a[m] = c * a_sel - s * b_sel
        b[m] = s * a_sel + c * b_sel
        out.update_quantity(a_key, a)
        out.update_quantity(b_key, b)
    return out
