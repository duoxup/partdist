from __future__ import annotations

from functools import partial
from typing import Callable, Iterable, Literal

import numpy as np

from ._array_helpers import as_1d_float_array as _as_1d_float_array_impl

ArrayLike = float | Iterable[float] | np.ndarray
NormMode = Literal["none", "peak", "area"]


def _as_1d_float_array(x: ArrayLike) -> np.ndarray:
    return _as_1d_float_array_impl(x, "x")


def _normalize_profile(
    x: np.ndarray,
    y: np.ndarray,
    *,
    mode: NormMode = "none",
    target: float = 1.0,
) -> np.ndarray:
    """
    Normalize a sampled profile.

    Parameters
    ----------
    x
        Sample positions.
    y
        Sampled profile values.
    mode
        - "none": no normalization
        - "peak": scale so that max(y) == target
        - "area": scale so that integral(y dx) == target
    target
        Target peak value or target area, depending on `mode`.

    Returns
    -------
    np.ndarray
        Normalized profile values.
    """
    y = np.asarray(y, dtype=float)

    if mode == "none":
        return y

    if mode == "peak":
        ymax = float(np.max(y))
        if ymax == 0.0:
            raise ValueError("Cannot normalize by peak because the profile peak is zero.")
        return y * (target / ymax)

    if mode == "area":
        area = float(np.trapezoid(y, x))
        if area == 0.0:
            raise ValueError("Cannot normalize by area because the profile area is zero.")
        return y * (target / area)

    raise ValueError(f"Unsupported normalization mode: {mode!r}")


def uniform_profile(
    x: ArrayLike,
    *,
    width: float,
    center: float = 0.0,
    amplitude: float = 1.0,
    outside_value: float = 0.0,
    normalize: NormMode = "none",
    target: float = 1.0,
) -> np.ndarray:
    """
    Uniform (flat-top) profile.

    Parameters
    ----------
    x
        Coordinate array.
    width
        Full width of the flat region.
    center
        Center position.
    amplitude
        Plateau value before optional normalization.
    outside_value
        Value outside the support region.
    normalize
        "none", "peak", or "area".
    target
        Target value for normalization.

    Returns
    -------
    np.ndarray
        Sampled profile values.
    """
    if width < 0.0:
        raise ValueError("width must be >= 0.")

    x = _as_1d_float_array(x)
    dx = x - center
    half_width = 0.5 * width

    y = np.full_like(x, outside_value, dtype=float)
    mask = np.abs(dx) <= half_width
    y[mask] = amplitude

    return _normalize_profile(x, y, mode=normalize, target=target)


def plateau_profile(
    x: ArrayLike,
    *,
    plateau_width: float,
    edge_width: float,
    center: float = 0.0,
    amplitude: float = 1.0,
    outside_value: float = 0.0,
    normalize: NormMode = "none",
    target: float = 1.0,
) -> np.ndarray:
    """
    Plateau profile with linear edges (trapezoidal profile).

    Shape:
        outside_value                  for |x-center| > plateau_width/2 + edge_width
        linear ramp                    in the edge regions
        amplitude                      for |x-center| <= plateau_width/2

    Parameters
    ----------
    x
        Coordinate array.
    plateau_width
        Full width of the flat central plateau.
    edge_width
        Width of each linear edge on both sides.
    center
        Center position.
    amplitude
        Plateau value before optional normalization.
    outside_value
        Value outside the support region.
    normalize
        "none", "peak", or "area".
    target
        Target value for normalization.

    Returns
    -------
    np.ndarray
        Sampled profile values.
    """
    if plateau_width < 0.0:
        raise ValueError("plateau_width must be >= 0.")
    if edge_width < 0.0:
        raise ValueError("edge_width must be >= 0.")

    x = _as_1d_float_array(x)
    dx = np.abs(x - center)

    half_plateau = 0.5 * plateau_width
    outer = half_plateau + edge_width

    y = np.full_like(x, outside_value, dtype=float)

    if edge_width == 0.0:
        mask = dx <= half_plateau
        y[mask] = amplitude
        return _normalize_profile(x, y, mode=normalize, target=target)

    mask_plateau = dx <= half_plateau
    mask_edge = (dx > half_plateau) & (dx <= outer)

    y[mask_plateau] = amplitude
    y[mask_edge] = amplitude * (1.0 - (dx[mask_edge] - half_plateau) / edge_width)

    return _normalize_profile(x, y, mode=normalize, target=target)


def inverted_parabola_profile(
    x: ArrayLike,
    *,
    half_width: float,
    center: float = 0.0,
    amplitude: float = 1.0,
    outside_value: float = 0.0,
    normalize: NormMode = "none",
    target: float = 1.0,
) -> np.ndarray:
    """
    Inverted parabolic profile.

    Shape:
        y = amplitude * (1 - ((x-center)/half_width)^2),    for |x-center| <= half_width
        y = outside_value,                                  otherwise

    Parameters
    ----------
    x
        Coordinate array.
    half_width
        Half-width of the support region.
    center
        Center position.
    amplitude
        Peak value before optional normalization.
    outside_value
        Value outside the support region.
    normalize
        "none", "peak", or "area".
    target
        Target value for normalization.

    Returns
    -------
    np.ndarray
        Sampled profile values.
    """
    if half_width < 0.0:
        raise ValueError("half_width must be >= 0.")

    x = _as_1d_float_array(x)
    dx = x - center

    y = np.full_like(x, outside_value, dtype=float)

    if half_width == 0.0:
        mask = dx == 0.0
        y[mask] = amplitude
        return _normalize_profile(x, y, mode=normalize, target=target)

    mask = np.abs(dx) <= half_width
    u = dx[mask] / half_width
    y[mask] = amplitude * (1.0 - u**2)

    return _normalize_profile(x, y, mode=normalize, target=target)


def gaussian_profile(
    x: ArrayLike,
    *,
    sigma: float,
    center: float = 0.0,
    amplitude: float = 1.0,
    truncate_at: float | None = None,
    outside_value: float = 0.0,
    normalize: NormMode = "none",
    target: float = 1.0,
) -> np.ndarray:
    """
    Gaussian profile, optionally truncated.

    Shape:
        y = amplitude * exp(-0.5 * ((x-center)/sigma)^2)

    If `truncate_at` is not None and > 0, the profile is replaced by
    `outside_value` for |x-center| > truncate_at * sigma.

    Special cases:
        - truncate_at is None: no truncation
        - truncate_at == 0: no truncation

    Parameters
    ----------
    x
        Coordinate array.
    sigma
        RMS width of the Gaussian.
    center
        Center position.
    amplitude
        Peak value before optional normalization.
    truncate_at
        Truncation threshold in units of sigma.
        Example: truncate_at=3 means keep only |x-center| <= 3*sigma.
        If None or 0, no truncation is applied.
    outside_value
        Value outside the support region when truncated.
    normalize
        "none", "peak", or "area".
    target
        Target value for normalization.

    Returns
    -------
    np.ndarray
        Sampled profile values.
    """
    if sigma <= 0.0:
        raise ValueError("sigma must be > 0.")
    if truncate_at is not None and truncate_at < 0.0:
        raise ValueError("truncate_at must be >= 0 or None.")

    x = _as_1d_float_array(x)
    dx = x - center

    y = amplitude * np.exp(-0.5 * (dx / sigma) ** 2)
    
    if truncate_at is not None and truncate_at > 0.0:
        mask_outside = np.abs(dx) > truncate_at * sigma
        y = np.where(mask_outside, outside_value, y)

    return _normalize_profile(x, y, mode=normalize, target=target)

def sample_profile(
    profile_func: Callable[[np.ndarray], np.ndarray],
    x: ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample a profile callable on the given coordinates.

    Parameters
    ----------
    profile_func
        Callable taking a numpy array and returning a numpy array.
    x
        Sample coordinates.

    Returns
    -------
    (x_sample, y_sample)
        Sampled curve arrays.
    """
    x_arr = _as_1d_float_array(x)
    y_arr = np.asarray(profile_func(x_arr), dtype=float)

    if y_arr.shape != x_arr.shape:
        raise ValueError("profile_func(x) must return an array with the same shape as x.")

    return x_arr, y_arr


def make_uniform_profile(**kwargs) -> Callable[[np.ndarray], np.ndarray]:
    """
    Return a callable uniform profile.
    """
    return partial(uniform_profile, **kwargs)


def make_plateau_profile(**kwargs) -> Callable[[np.ndarray], np.ndarray]:
    """
    Return a callable plateau profile.
    """
    return partial(plateau_profile, **kwargs)


def make_inverted_parabola_profile(**kwargs) -> Callable[[np.ndarray], np.ndarray]:
    """
    Return a callable inverted parabolic profile.
    """
    return partial(inverted_parabola_profile, **kwargs)


def make_gaussian_profile(**kwargs) -> Callable[[np.ndarray], np.ndarray]:
    """
    Return a callable Gaussian profile.
    """
    return partial(gaussian_profile, **kwargs)