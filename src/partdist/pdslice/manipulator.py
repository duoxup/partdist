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


def center_beam(
    dist: SliceDistribution,
    *,
    axes: Sequence[str] = ("x", "y", "px", "py"),
    weight: Union[None, str, ArrayLike] = "lam_abs",
    mask: Optional[Union[np.ndarray, Sequence[bool]]] = None,
    inplace: bool = False,
) -> SliceDistribution:
    """Subtract the weighted mean from each axis in `axes` so its centroid → 0.

    Default `axes` excludes `pz` to preserve the longitudinal anchor that
    generator.make_slice requires for forward-beam consistency. Pass
    `axes=("x", "y", "px", "py", "pz")` explicitly to center pz as well.
    """
    out = _copy_or_inplace(dist, inplace=inplace)
    n = len(out)
    m = _normalize_mask(mask, n)
    w = _get_weight_array(out, weight, absolute=True)
    valid = m & np.isfinite(w) & (w > 0.0)
    wsum = float(np.sum(w[valid]))
    if wsum <= 0.0:
        raise ValueError("Total weight in selected mask must be positive.")
    for key in axes:
        arr = out.get_data(key).copy()
        mean = float(np.sum(arr[valid] * w[valid]) / wsum)
        arr[m] -= mean
        out.update_quantity(key, arr)
    return out


def match_twiss_x(
    dist: SliceDistribution,
    alpha: float,
    beta: float,
    *,
    weight: Union[None, str, ArrayLike] = "lam_abs",
    mask: Optional[Union[np.ndarray, Sequence[bool]]] = None,
    center_before_match: bool = True,
    preserve_centroid: bool = True,
    inplace: bool = False,
) -> SliceDistribution:
    """Match Twiss (alpha, beta) in the x plane via Courant-Snyder.

    Applies the Courant-Snyder transformation that maps the current
    weighted Twiss parameters in the x plane to the target (alpha, beta),
    preserving geometric emittance exactly.

    Parameters
    ----------
    dist : SliceDistribution
        Input distribution.
    alpha : float
        Target Twiss alpha parameter.
    beta : float
        Target Twiss beta parameter [m/rad].  Must be positive.
    weight : str or array-like or None
        Particle weights for second-moment computation.
        Defaults to ``"lam_abs"`` (charge-weighted).
    mask : array-like of bool or None
        Boolean mask selecting particles to include in the Twiss
        computation and transformation.  Unmasked particles are
        left unchanged.
    center_before_match : bool
        If True (default), use centered covariance (subtract the
        weighted mean before computing current Twiss).  If False,
        use raw second moments about zero.
    preserve_centroid : bool
        Only active when ``center_before_match=True``.  If True
        (default), the weighted mean of x and x' is restored after
        the transformation so the centroid position is unchanged.
    inplace : bool
        Modify ``dist`` in place when True.  Returns the modified
        distribution either way.

    Returns
    -------
    SliceDistribution
        Distribution with x (and px) updated to match target Twiss.
        y, py, pz, and all other quantities are unchanged.
    """
    if beta <= 0.0:
        raise ValueError(f"Target beta must be positive, got {beta}.")

    out = _copy_or_inplace(dist, inplace=inplace)
    n = len(out)
    w = _get_weight_array(out, weight, absolute=True)
    m = _normalize_mask(mask, n)

    u = _extract_data(out, "x", n_expected=n, dtype=float, name="x")
    pu = _extract_data(out, "px", n_expected=n, dtype=float, name="px")
    pz = _extract_data(out, "pz", n_expected=n, dtype=float, name="pz")

    valid = (
        m
        & np.isfinite(u)
        & np.isfinite(pu)
        & np.isfinite(pz)
        & np.isfinite(w)
        & (np.abs(pz) > 0.0)
    )
    if np.count_nonzero(valid) < 2:
        raise ValueError("At least two valid selected particles are required for Twiss matching.")
    if float(np.sum(w[valid])) <= 0.0:
        raise ValueError("Selected particles must have strictly positive total weight.")

    up = pu / pz
    u_sel, up_sel, pz_sel, w_sel = u[valid], up[valid], pz[valid], w[valid]

    if center_before_match:
        alpha_old, beta_old, _eps_old, mean_u, mean_up = (
            _weighted_centered_twiss_from_arrays(u_sel, up_sel, w_sel)
        )
        work_u = u_sel - mean_u
        work_up = up_sel - mean_up
    else:
        alpha_old, beta_old, _eps_old = _weighted_raw_twiss_from_arrays(u_sel, up_sel, w_sel)
        mean_u = 0.0
        mean_up = 0.0
        work_u = u_sel
        work_up = up_sel

    if beta_old <= 0.0:
        raise ValueError("Current beta is non-positive, cannot perform Twiss matching.")

    # Courant-Snyder transformation matrix (upper-triangular; r12 ≡ 0)
    r11 = np.sqrt(beta / beta_old)
    r21 = (alpha_old - alpha) / np.sqrt(beta_old * beta)
    r22 = np.sqrt(beta_old / beta)

    work_u_new = r11 * work_u
    work_up_new = r21 * work_u + r22 * work_up

    if center_before_match and preserve_centroid:
        u_new_sel = work_u_new + mean_u
        up_new_sel = work_up_new + mean_up
    else:
        u_new_sel = work_u_new
        up_new_sel = work_up_new

    u_new = u.copy()
    pu_new = pu.copy()
    u_new[valid] = u_new_sel
    pu_new[valid] = up_new_sel * pz_sel

    out.update_quantity("x", u_new)
    out.update_quantity("px", pu_new)
    return out


def match_twiss_y(
    dist: SliceDistribution,
    alpha: float,
    beta: float,
    *,
    weight: Union[None, str, ArrayLike] = "lam_abs",
    mask: Optional[Union[np.ndarray, Sequence[bool]]] = None,
    center_before_match: bool = True,
    preserve_centroid: bool = True,
    inplace: bool = False,
) -> SliceDistribution:
    """Match Twiss (alpha, beta) in the y plane via Courant-Snyder.

    Applies the Courant-Snyder transformation that maps the current
    weighted Twiss parameters in the y plane to the target (alpha, beta),
    preserving geometric emittance exactly.

    Parameters
    ----------
    dist : SliceDistribution
        Input distribution.
    alpha : float
        Target Twiss alpha parameter.
    beta : float
        Target Twiss beta parameter [m/rad].  Must be positive.
    weight : str or array-like or None
        Particle weights for second-moment computation.
        Defaults to ``"lam_abs"`` (charge-weighted).
    mask : array-like of bool or None
        Boolean mask selecting particles to include in the Twiss
        computation and transformation.  Unmasked particles are
        left unchanged.
    center_before_match : bool
        If True (default), use centered covariance (subtract the
        weighted mean before computing current Twiss).  If False,
        use raw second moments about zero.
    preserve_centroid : bool
        Only active when ``center_before_match=True``.  If True
        (default), the weighted mean of y and y' is restored after
        the transformation so the centroid position is unchanged.
    inplace : bool
        Modify ``dist`` in place when True.  Returns the modified
        distribution either way.

    Returns
    -------
    SliceDistribution
        Distribution with y (and py) updated to match target Twiss.
        x, px, pz, and all other quantities are unchanged.
    """
    if beta <= 0.0:
        raise ValueError(f"Target beta must be positive, got {beta}.")

    out = _copy_or_inplace(dist, inplace=inplace)
    n = len(out)
    w = _get_weight_array(out, weight, absolute=True)
    m = _normalize_mask(mask, n)

    u = _extract_data(out, "y", n_expected=n, dtype=float, name="y")
    pu = _extract_data(out, "py", n_expected=n, dtype=float, name="py")
    pz = _extract_data(out, "pz", n_expected=n, dtype=float, name="pz")

    valid = (
        m
        & np.isfinite(u)
        & np.isfinite(pu)
        & np.isfinite(pz)
        & np.isfinite(w)
        & (np.abs(pz) > 0.0)
    )
    if np.count_nonzero(valid) < 2:
        raise ValueError("At least two valid selected particles are required for Twiss matching.")
    if float(np.sum(w[valid])) <= 0.0:
        raise ValueError("Selected particles must have strictly positive total weight.")

    up = pu / pz
    u_sel, up_sel, pz_sel, w_sel = u[valid], up[valid], pz[valid], w[valid]

    if center_before_match:
        alpha_old, beta_old, _eps_old, mean_u, mean_up = (
            _weighted_centered_twiss_from_arrays(u_sel, up_sel, w_sel)
        )
        work_u = u_sel - mean_u
        work_up = up_sel - mean_up
    else:
        alpha_old, beta_old, _eps_old = _weighted_raw_twiss_from_arrays(u_sel, up_sel, w_sel)
        mean_u = 0.0
        mean_up = 0.0
        work_u = u_sel
        work_up = up_sel

    if beta_old <= 0.0:
        raise ValueError("Current beta is non-positive, cannot perform Twiss matching.")

    # Courant-Snyder transformation matrix (upper-triangular; r12 ≡ 0)
    r11 = np.sqrt(beta / beta_old)
    r21 = (alpha_old - alpha) / np.sqrt(beta_old * beta)
    r22 = np.sqrt(beta_old / beta)

    work_u_new = r11 * work_u
    work_up_new = r21 * work_u + r22 * work_up

    if center_before_match and preserve_centroid:
        u_new_sel = work_u_new + mean_u
        up_new_sel = work_up_new + mean_up
    else:
        u_new_sel = work_u_new
        up_new_sel = work_up_new

    u_new = u.copy()
    pu_new = pu.copy()
    u_new[valid] = u_new_sel
    pu_new[valid] = up_new_sel * pz_sel

    out.update_quantity("y", u_new)
    out.update_quantity("py", pu_new)
    return out


def match_twiss_xy(
    dist: SliceDistribution,
    alpha_x: float,
    beta_x: float,
    alpha_y: float,
    beta_y: float,
    *,
    weight: Union[None, str, ArrayLike] = "lam_abs",
    mask: Optional[Union[np.ndarray, Sequence[bool]]] = None,
    center_before_match: bool = True,
    preserve_centroid: bool = True,
    inplace: bool = False,
) -> SliceDistribution:
    """Match Twiss in both transverse planes (x then y)."""
    out = match_twiss_x(
        dist,
        alpha=alpha_x, beta=beta_x,
        weight=weight, mask=mask,
        center_before_match=center_before_match,
        preserve_centroid=preserve_centroid,
        inplace=inplace,
    )
    return match_twiss_y(
        out,
        alpha=alpha_y, beta=beta_y,
        weight=weight, mask=mask,
        center_before_match=center_before_match,
        preserve_centroid=preserve_centroid,
        inplace=True,
    )


def apply_dispersion(
    dist: SliceDistribution,
    D: float,
    *,
    axis: Literal["x", "y"] = "x",
    p_ref: Optional[float] = None,
    weight: Union[None, str, ArrayLike] = "lam_abs",
    mask: Optional[Union[np.ndarray, Sequence[bool]]] = None,
    inplace: bool = False,
) -> SliceDistribution:
    """Add position dispersion: pos ← pos + D · (pz - p_ref) / p_ref."""
    if axis not in {"x", "y"}:
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}.")

    out = _copy_or_inplace(dist, inplace=inplace)
    n = len(out)
    m = _normalize_mask(mask, n)
    pz = out.get_data("pz")

    if p_ref is None:
        w = _get_weight_array(out, weight, absolute=True)
        valid = m & np.isfinite(w) & (w > 0.0) & np.isfinite(pz)
        wsum = float(np.sum(w[valid]))
        if wsum <= 0.0:
            raise ValueError("Total weight must be positive to derive p_ref.")
        p_ref = float(np.sum(pz[valid] * w[valid]) / wsum)

    if p_ref == 0.0:
        raise ValueError("p_ref must be non-zero.")

    delta = (pz - p_ref) / p_ref
    pos = out.get_data(axis).copy()
    pos[m] += D * delta[m]
    out.update_quantity(axis, pos)
    return out


def scale_rms_x(
    dist: SliceDistribution,
    *,
    factor: float,
    emittance_preserving: bool = True,
    mask: Optional[Union[np.ndarray, Sequence[bool]]] = None,
    inplace: bool = False,
) -> SliceDistribution:
    """Scale x by `factor` (and px by 1/factor if emittance-preserving).

    Parameters
    ----------
    dist : SliceDistribution
        Input distribution.
    factor : float
        Scale factor for x (must be positive).
    emittance_preserving : bool
        If True (default), also scale px by 1/factor to preserve geometric
        emittance. If False, px is left unchanged.
    mask : array-like of bool or None
        Boolean mask selecting particles to transform.  Unmasked particles
        are left unchanged.
    inplace : bool
        Modify ``dist`` in place when True.  Returns the modified
        distribution either way.

    Returns
    -------
    SliceDistribution
        Distribution with x (and optionally px) updated.
        y, py, pz, and all other quantities are unchanged.
    """
    if factor <= 0.0:
        raise ValueError(f"factor must be > 0, got {factor}.")
    out = _copy_or_inplace(dist, inplace=inplace)
    n = len(out)
    m = _normalize_mask(mask, n)
    x = out.get_data("x").copy()
    x[m] *= factor
    out.update_quantity("x", x)
    if emittance_preserving:
        px = out.get_data("px").copy()
        px[m] /= factor
        out.update_quantity("px", px)
    return out


def scale_rms_y(
    dist: SliceDistribution,
    *,
    factor: float,
    emittance_preserving: bool = True,
    mask: Optional[Union[np.ndarray, Sequence[bool]]] = None,
    inplace: bool = False,
) -> SliceDistribution:
    """Scale y by `factor` (and py by 1/factor if emittance-preserving).

    Parameters
    ----------
    dist : SliceDistribution
        Input distribution.
    factor : float
        Scale factor for y (must be positive).
    emittance_preserving : bool
        If True (default), also scale py by 1/factor to preserve geometric
        emittance. If False, py is left unchanged.
    mask : array-like of bool or None
        Boolean mask selecting particles to transform.  Unmasked particles
        are left unchanged.
    inplace : bool
        Modify ``dist`` in place when True.  Returns the modified
        distribution either way.

    Returns
    -------
    SliceDistribution
        Distribution with y (and optionally py) updated.
        x, px, pz, and all other quantities are unchanged.
    """
    if factor <= 0.0:
        raise ValueError(f"factor must be > 0, got {factor}.")
    out = _copy_or_inplace(dist, inplace=inplace)
    n = len(out)
    m = _normalize_mask(mask, n)
    y = out.get_data("y").copy()
    y[m] *= factor
    out.update_quantity("y", y)
    if emittance_preserving:
        py = out.get_data("py").copy()
        py[m] /= factor
        out.update_quantity("py", py)
    return out
