from __future__ import annotations

from typing import Callable, Literal, Optional, Sequence, Union

import numpy as np

from scipy.optimize import brentq
from xtils import g_c, g_e0, g_m0
from xtils import relconv

from..particle_array_quantity import ParticleArrayQuantity
from .core import ParticleDistribution
from .analysis import (TrendFitResult,
                       fit_trend,
                       fit_linear_chirp,
                       # compute_twiss_plane,
                       # compute_phase_space_plane,
                       )
from .utils import (
    _as_1d_array,
    _copy_or_inplace,
    _extract_data,
    _get_weight_array,
    _normalize_mask,
    _replace_velocity_from_momentum,
    _update_momentum_components
)



ArrayLike = Union[np.ndarray, Sequence[float], float]
CurveLike = Union[
    Callable[[np.ndarray], ArrayLike],
    tuple[ArrayLike, ArrayLike],
]
ResidualMode = Literal["preserve", "discard", "rescale"]


def _weighted_mean_1d(values: np.ndarray, weights: np.ndarray) -> float:
    wsum = float(np.sum(weights))
    if wsum <= 0.0:
        raise ValueError("Total weight must be positive.")
    return float(np.sum(weights * values) / wsum)


def _weighted_mean_kinetic_energy_eV_from_momentum(
    px: np.ndarray,
    py: np.ndarray,
    pz: np.ndarray,
    weights: np.ndarray,
    *,
    m0: float = g_m0,
    q: float = g_e0,
    c: float = g_c,
) -> float:
    """
    Compute the weighted mean kinetic energy from momentum components [eV/c].
    """
    p_abs = np.sqrt(px**2 + py**2 + pz**2)
    ke_eV = np.asarray(relconv.ke_eV_from_p_eVc(p_abs, m0=m0, q=q, c=c), dtype=float)
    return _weighted_mean_1d(ke_eV, weights)

def _build_twiss_covariance(alpha: float, beta: float, emittance: float) -> np.ndarray:
    """
    Build the 2x2 covariance matrix from Twiss parameters and emittance.
    """
    if beta <= 0.0:
        raise ValueError("Target beta must be positive.")
    if emittance <= 0.0:
        raise ValueError("Emittance must be positive.")

    gamma_twiss = (1.0 + alpha**2) / beta

    return emittance * np.array(
        [
            [beta, -alpha],
            [-alpha, gamma_twiss],
        ],
        dtype=float,
    )

def _evaluate_trend_like(
    x: np.ndarray,
    trend: Union[TrendFitResult, Callable[[ArrayLike], ArrayLike]],
) -> np.ndarray:
    """
    Evaluate a trend-like object on x and return a 1D float array.
    """
    if isinstance(trend, TrendFitResult):
        y = trend(x)
    else:
        y = trend(x)

    return np.asarray(y, dtype=float).reshape(-1)


def _build_residual_scaled_array(
    original: np.ndarray,
    original_trend: np.ndarray,
    target_trend: np.ndarray,
    *,
    residual_mode: ResidualMode,
    residual_scale: float = 1.0,
) -> np.ndarray:
    """
    Build a new quantity array from an original array, original trend,
    and target trend according to the selected residual policy.

    Parameters
    ----------
    original : np.ndarray
        Original particle quantity values.
    original_trend : np.ndarray
        Original fitted/evaluated trend at each particle x.
    target_trend : np.ndarray
        Target trend at each particle x.
    residual_mode : {"preserve", "discard", "rescale"}
        Residual handling mode.
    residual_scale : float
        Residual scaling factor used only when residual_mode="rescale".

    Returns
    -------
    np.ndarray
        New quantity array.
    """
    residual = original - original_trend

    if residual_mode == "preserve":
        return target_trend + residual

    if residual_mode == "discard":
        return target_trend

    if residual_mode == "rescale":
        return target_trend + residual_scale * residual

    raise ValueError(f"Unknown residual_mode={residual_mode!r}.")


def _update_quantity_array(
    dist: "ParticleDistribution",
    key: str,
    values: ArrayLike,
    *,
    m0: float = g_m0,
    q: float = g_e0,
    c: float = g_c,
    inplace: bool = False,
) -> "ParticleDistribution":
    """
    Update a quantity in a ParticleDistribution.

    For base and extra quantities, calls dist.update_quantity(key, values).
    Derived quantities (other than px/py/pz, which are now base) are not
    directly writable.

    Parameters
    ----------
    dist : ParticleDistribution
        Input distribution.
    key : str
        Quantity key to update.
    values : array-like
        New particle-resolved values.
    m0 : float
        Rest mass in [kg].
    q : float
        Charge in [C].
    c : float
        Speed of light in [m/s].
    inplace : bool
        Whether to modify the input object directly.

    Returns
    -------
    ParticleDistribution
        Updated distribution.
    """
    out = _copy_or_inplace(dist, inplace=inplace)
    n = len(out)

    arr = _as_1d_array(values, dtype=float, name=key)
    if len(arr) != n:
        raise ValueError(f"{key} must have length {n}, got {len(arr)}.")

    kind = out.quantity_kind(key)

    if kind in {"base", "extra"}:
        out.update_quantity(key, arr)
        return out

    raise ValueError(
        f"Derived quantity {key!r} is not directly writable. "
        "Update the underlying base quantities (px, py, pz) instead."
    )

def _weighted_raw_twiss_from_arrays(
    u: np.ndarray,
    up: np.ndarray,
    w: np.ndarray,
) -> tuple[float, float, float]:
    """
    Compute (alpha, beta, geometric_emittance) from raw weighted second moments:
        <u^2>, <u u'>, <u'^2>
    """
    wsum = float(np.sum(w))
    if wsum <= 0.0:
        raise ValueError("Total weight must be positive.")

    s11 = float(np.sum(w * u * u) / wsum)
    s12 = float(np.sum(w * u * up) / wsum)
    s22 = float(np.sum(w * up * up) / wsum)

    det = s11 * s22 - s12**2
    if det <= 0.0:
        raise ValueError("Raw second-moment determinant must be positive.")

    eps = float(np.sqrt(det))
    beta = float(s11 / eps)
    alpha = float(-s12 / eps)

    return alpha, beta, eps


def _weighted_centered_twiss_from_arrays(
    u: np.ndarray,
    up: np.ndarray,
    w: np.ndarray,
) -> tuple[float, float, float, float, float]:
    """
    Compute centered Twiss parameters from weighted covariance.

    Returns
    -------
    alpha, beta, geometric_emittance, mean_u, mean_up
    """
    wsum = float(np.sum(w))
    if wsum <= 0.0:
        raise ValueError("Total weight must be positive.")

    mean_u = float(np.sum(w * u) / wsum)
    mean_up = float(np.sum(w * up) / wsum)

    du = u - mean_u
    dup = up - mean_up

    s11 = float(np.sum(w * du * du) / wsum)
    s12 = float(np.sum(w * du * dup) / wsum)
    s22 = float(np.sum(w * dup * dup) / wsum)

    det = s11 * s22 - s12**2
    if det <= 0.0:
        raise ValueError("Centered second-moment determinant must be positive.")

    eps = float(np.sqrt(det))
    beta = float(s11 / eps)
    alpha = float(-s12 / eps)

    return alpha, beta, eps, mean_u, mean_up


def _infer_reference_velocity(dist: ParticleDistribution) -> float:
    """
    Infer a reference longitudinal velocity from the distribution.

    Uses |Q| as weights when possible; falls back to a simple mean otherwise.
    """
    vz = np.asarray(dist.vz, dtype=float).reshape(-1)
    q_abs = np.abs(np.asarray(dist.Q, dtype=float).reshape(-1))

    if len(vz) == 0:
        raise ValueError("Cannot infer reference velocity from an empty distribution.")

    if np.sum(q_abs) > 0.0:
        v_ref = float(np.sum(q_abs * vz) / np.sum(q_abs))
    else:
        v_ref = float(np.mean(vz))

    if v_ref == 0.0:
        raise ValueError("Inferred reference velocity is zero.")
    return v_ref

def _evaluate_curve_like(
    x: np.ndarray,
    curve: CurveLike,
    *,
    outside_value: float = 0.0,
) -> np.ndarray:
    """
    Evaluate a curve-like input on x.

    Supported curve formats
    -----------------------
    1. callable:
        curve(x_eval) -> array-like
    2. sampled tuple:
        curve = (x_sample, y_sample)

    Parameters
    ----------
    x : np.ndarray
        Evaluation points.
    curve : callable or (x_sample, y_sample)
        Curve specification.
    outside_value : float
        Value used outside the sampled x range when curve is given as arrays.

    Returns
    -------
    np.ndarray
        Evaluated multiplicative factors.
    """
    if callable(curve):
        y = np.asarray(curve(x), dtype=float).reshape(-1)
        if y.size == 1:
            y = np.full_like(x, float(y[0]), dtype=float)
        if len(y) != len(x):
            raise ValueError(
                f"Callable curve returned length {len(y)}, expected {len(x)}."
            )
        return y

    if isinstance(curve, tuple) and len(curve) == 2:
        x_sample = np.asarray(curve[0], dtype=float).reshape(-1)
        y_sample = np.asarray(curve[1], dtype=float).reshape(-1)

        if len(x_sample) != len(y_sample):
            raise ValueError("curve sample arrays must have the same length.")
        if len(x_sample) < 2:
            raise ValueError("curve sample arrays must contain at least two points.")

        order = np.argsort(x_sample, kind="stable")
        x_sorted = x_sample[order]
        y_sorted = y_sample[order]

        return np.interp(
            x,
            x_sorted,
            y_sorted,
            left=outside_value,
            right=outside_value,
        )

    raise TypeError(
        "curve must be either a callable or a tuple (x_sample, y_sample)."
    )
    
def apply_trend_delta(
    dist: "ParticleDistribution",
    *,
    x_key: str,
    y_key: str,
    delta_trend: Union[TrendFitResult, Callable[[ArrayLike], ArrayLike]],
    mask: Optional[Union[np.ndarray, Sequence[bool]]] = None,
    m0: float = g_m0,
    q: float = g_e0,
    c: float = g_c,
    inplace: bool = False,
) -> "ParticleDistribution":
    """
    Add a trend delta to y as a function of x.

    This implements:
        y_new = y_old + delta(x)

    Parameters
    ----------
    dist : ParticleDistribution
        Input distribution.
    x_key : str
        Independent-variable quantity key.
    y_key : str
        Dependent-variable quantity key to modify.
    delta_trend : TrendFitResult or callable
        Delta trend evaluated as delta(x).
    mask : array-like of bool, optional
        Mask selecting particles to modify. Unmasked particles remain unchanged.
    m0 : float
        Rest mass in [kg].
    q : float
        Charge in [C].
    c : float
        Speed of light in [m/s].
    inplace : bool
        Whether to modify the input object directly.

    Returns
    -------
    ParticleDistribution
        Modified distribution.
    """
    n = len(dist)
    x = _extract_data(dist, x_key, n_expected=n, dtype=float, name=x_key)
    y = _extract_data(dist, y_key, n_expected=n, dtype=float, name=y_key)
    m = _normalize_mask(mask, n)

    y_new = y.copy()
    y_new[m] = y[m] + _evaluate_trend_like(x[m], delta_trend)

    return _update_quantity_array(
        dist,
        y_key,
        y_new,
        m0=m0,
        q=q,
        c=c,
        inplace=inplace,
    )


def replace_trend(
    dist: "ParticleDistribution",
    *,
    x_key: str,
    y_key: str,
    target_trend: Union[TrendFitResult, Callable[[ArrayLike], ArrayLike]],
    original_trend: Optional[Union[TrendFitResult, Callable[[ArrayLike], ArrayLike]]] = None,
    fit_original_if_missing: bool = True,
    fit_bins: int = 100,
    fit_weight: Union[None, str, ArrayLike] = None,
    fit_stat: Literal["mean", "weighted_mean", "median"] = "weighted_mean",
    fit_min_count: int = 1,
    fit_min_weight_sum: float = 0.0,
    fit_method: Literal["linear", "poly", "spline"] = "spline",
    fit_poly_order: int = 1,
    fit_spline_smoothing: float = 0.0,
    fit_spline_weight: Literal["count", "sqrt_count", "weight_sum", "sqrt_weight_sum", "none"] = "sqrt_count",
    residual_mode: ResidualMode = "preserve",
    residual_scale: float = 1.0,
    mask: Optional[Union[np.ndarray, Sequence[bool]]] = None,
    m0: float = g_m0,
    q: float = g_e0,
    c: float = g_c,
    inplace: bool = False,
) -> "ParticleDistribution":
    """
    Replace the y(x) trend by a target trend.

    The modified quantity is constructed as:
        y_new = target_trend(x) + residual_policy(y_old - original_trend(x))

    depending on residual_mode.

    Parameters
    ----------
    dist : ParticleDistribution
        Input distribution.
    x_key : str
        Independent-variable quantity key.
    y_key : str
        Dependent-variable quantity key to modify.
    target_trend : TrendFitResult or callable
        Target trend evaluated as f_target(x).
    original_trend : TrendFitResult or callable, optional
        Original trend f_old(x). If omitted and fit_original_if_missing=True,
        it is fitted automatically from the current distribution.
    fit_original_if_missing : bool
        Whether to fit the original trend automatically if original_trend is None.
    fit_bins, fit_weight, fit_stat, fit_min_count, fit_min_weight_sum, fit_method,
    fit_poly_order, fit_spline_smoothing, fit_spline_weight :
        Parameters used when fitting the original trend automatically.
    residual_mode : {"preserve", "discard", "rescale"}
        Residual handling mode.
    residual_scale : float
        Residual scaling factor used only when residual_mode="rescale".
    mask : array-like of bool, optional
        Mask selecting particles to modify. Unmasked particles remain unchanged.
    m0 : float
        Rest mass in [kg].
    q : float
        Charge in [C].
    c : float
        Speed of light in [m/s].
    inplace : bool
        Whether to modify the input object directly.

    Returns
    -------
    ParticleDistribution
        Modified distribution.
    """
    n = len(dist)
    x = _extract_data(dist, x_key, n_expected=n, dtype=float, name=x_key)
    y = _extract_data(dist, y_key, n_expected=n, dtype=float, name=y_key)
    m = _normalize_mask(mask, n)

    if original_trend is None:
        if not fit_original_if_missing:
            raise ValueError(
                "original_trend is None and fit_original_if_missing=False."
            )
        original_trend = fit_trend(
            dist,
            x_key,
            y_key,
            bins=fit_bins,
            weight=fit_weight,
            stat=fit_stat,
            min_count=fit_min_count,
            min_weight_sum=fit_min_weight_sum,
            mask=mask,
            method=fit_method,
            poly_order=fit_poly_order,
            spline_smoothing=fit_spline_smoothing,
            spline_weight=fit_spline_weight,
        )

    y_new = y.copy()

    x_sel = x[m]
    y_sel = y[m]
    old_sel = _evaluate_trend_like(x_sel, original_trend)
    target_sel = _evaluate_trend_like(x_sel, target_trend)

    y_new[m] = _build_residual_scaled_array(
        y_sel,
        old_sel,
        target_sel,
        residual_mode=residual_mode,
        residual_scale=residual_scale,
    )

    return _update_quantity_array(
        dist,
        y_key,
        y_new,
        m0=m0,
        q=q,
        c=c,
        inplace=inplace,
    )


def apply_pz_delta(
    dist: "ParticleDistribution",
    delta_pz: Union[TrendFitResult, Callable[[ArrayLike], ArrayLike]],
    *,
    z_key: str = "z",
    pz_key: str = "pz",
    mask: Optional[Union[np.ndarray, Sequence[bool]]] = None,
    m0: float = g_m0,
    q: float = g_e0,
    c: float = g_c,
    inplace: bool = False,
) -> "ParticleDistribution":
    """
    Add a momentum chirp delta to pz as a function of z.

    This implements:
        pz_new = pz_old + delta_pz(z)
    """
    return apply_trend_delta(
        dist,
        x_key=z_key,
        y_key=pz_key,
        delta_trend=delta_pz,
        mask=mask,
        m0=m0,
        q=q,
        c=c,
        inplace=inplace,
    )


def set_pz_trend(
    dist: "ParticleDistribution",
    target_trend: Union[TrendFitResult, Callable[[ArrayLike], ArrayLike]],
    *,
    z_key: str = "z",
    pz_key: str = "pz",
    original_trend: Optional[Union[TrendFitResult, Callable[[ArrayLike], ArrayLike]]] = None,
    fit_original_if_missing: bool = True,
    fit_bins: int = 100,
    fit_weight: Union[None, str, ArrayLike] = None,
    fit_stat: Literal["mean", "weighted_mean", "median"] = "weighted_mean",
    fit_min_count: int = 1,
    fit_min_weight_sum: float = 0.0,
    fit_method: Literal["linear", "poly", "spline"] = "spline",
    fit_poly_order: int = 1,
    fit_spline_smoothing: float = 0.0,
    fit_spline_weight: Literal["count", "sqrt_count", "weight_sum", "sqrt_weight_sum", "none"] = "sqrt_count",
    residual_mode: ResidualMode = "preserve",
    residual_scale: float = 1.0,
    mask: Optional[Union[np.ndarray, Sequence[bool]]] = None,
    m0: float = g_m0,
    q: float = g_e0,
    c: float = g_c,
    inplace: bool = False,
) -> "ParticleDistribution":
    """
    Replace the longitudinal trend pz(z) by a target trend.

    Parameters are the same as replace_trend(...), specialized to x=z and y=pz.
    """
    return replace_trend(
        dist,
        x_key=z_key,
        y_key=pz_key,
        target_trend=target_trend,
        original_trend=original_trend,
        fit_original_if_missing=fit_original_if_missing,
        fit_bins=fit_bins,
        fit_weight=fit_weight,
        fit_stat=fit_stat,
        fit_min_count=fit_min_count,
        fit_min_weight_sum=fit_min_weight_sum,
        fit_method=fit_method,
        fit_poly_order=fit_poly_order,
        fit_spline_smoothing=fit_spline_smoothing,
        fit_spline_weight=fit_spline_weight,
        residual_mode=residual_mode,
        residual_scale=residual_scale,
        mask=mask,
        m0=m0,
        q=q,
        c=c,
        inplace=inplace,
    )

def set_linear_chirp(
    dist,
    slope: float,
    *,
    intercept: Optional[float] = None,
    z_key: str = "z",
    pz_key: str = "pz",
    center_x: bool = False,
    center_y: bool = False,
    residual_mode: ResidualMode = "preserve",
    residual_scale: float = 1.0,
    original_trend: Optional[Union[TrendFitResult, Callable[[ArrayLike], ArrayLike]]] = None,
    fit_original_if_missing: bool = True,
    fit_bins: int = 100,
    fit_weight: Union[None, str, ArrayLike] = None,
    fit_stat: Literal["mean", "weighted_mean", "median"] = "weighted_mean",
    fit_min_count: int = 1,
    fit_min_weight_sum: float = 0.0,
    fit_method: Literal["linear", "poly", "spline"] = "spline",
    fit_poly_order: int = 1,
    fit_spline_smoothing: float = 0.0,
    fit_spline_weight: Literal["count", "sqrt_count", "weight_sum", "sqrt_weight_sum", "none"] = "sqrt_count",
    mask: Optional[Union[np.ndarray, Sequence[bool]]] = None,
    weight_for_centroid: Union[None, str, ArrayLike] = "Q",
    preserve_mean_kinetic_energy: bool = True,
    weight_for_energy: Union[None, str, ArrayLike] = "Q",
    m0: float = g_m0,
    q: float = g_e0,
    c: float = g_c,
    inplace: bool = False,
):
    """
    Set a linear longitudinal chirp:
        pz_target(z) = slope * z + intercept

    with optional x/y centering and optional preservation of the weighted mean
    kinetic energy.

    Notes
    -----
    - center_x controls whether z is centered before applying the linear term
    - center_y controls whether the baseline is initially anchored near the
      weighted mean pz
    - preserve_mean_kinetic_energy solves an additional scalar offset so that
      the weighted mean kinetic energy is preserved after the chirp operation

    If intercept is explicitly provided and preserve_mean_kinetic_energy=True,
    an additional offset may still be applied in order to preserve the mean
    kinetic energy. If you need the intercept to be enforced exactly, set
    preserve_mean_kinetic_energy=False.
    """
    out = _copy_or_inplace(dist, inplace=inplace)
    n = len(out)

    z = _extract_data(out, z_key, n_expected=n, dtype=float, name=z_key)
    pz = _extract_data(out, pz_key, n_expected=n, dtype=float, name=pz_key)
    px = _extract_data(out, "px", n_expected=n, dtype=float, name="px")
    py = _extract_data(out, "py", n_expected=n, dtype=float, name="py")

    m = _normalize_mask(mask, n)
    if np.count_nonzero(m) == 0:
        raise ValueError("No particles selected by mask.")

    # Weights for centering
    w_cent = _get_weight_array(out, weight_for_centroid, absolute=True)
    if np.sum(w_cent[m]) <= 0.0:
        raise ValueError("Centering requires strictly positive selected total weight.")

    z_ref = float(np.sum(w_cent[m] * z[m]) / np.sum(w_cent[m])) if center_x else 0.0
    pz_ref = float(np.sum(w_cent[m] * pz[m]) / np.sum(w_cent[m])) if center_y else 0.0

    base_intercept = (0.0 if intercept is None else float(intercept)) + pz_ref

    # Determine the original trend used for residual handling
    if original_trend is None:
        if not fit_original_if_missing:
            raise ValueError(
                "original_trend is None and fit_original_if_missing=False."
            )
        original_trend = fit_trend(
            out,
            z_key,
            pz_key,
            bins=fit_bins,
            weight=fit_weight,
            stat=fit_stat,
            min_count=fit_min_count,
            min_weight_sum=fit_min_weight_sum,
            mask=mask,
            method=fit_method,
            poly_order=fit_poly_order,
            spline_smoothing=fit_spline_smoothing,
            spline_weight=fit_spline_weight,
        )

    z_sel = z[m]
    pz_sel = pz[m]
    old_trend_sel = _evaluate_trend_like(z_sel, original_trend)

    def _build_full_pz(delta_offset: float) -> np.ndarray:
        """
        Build the full pz array for a trial extra offset.
        """
        target_sel = slope * (z_sel - z_ref) + base_intercept + delta_offset

        new_sel = _build_residual_scaled_array(
            pz_sel,
            old_trend_sel,
            target_sel,
            residual_mode=residual_mode,
            residual_scale=residual_scale,
        )

        pz_full = pz.copy()
        pz_full[m] = new_sel
        return pz_full

    delta_offset = 0.0

    if preserve_mean_kinetic_energy:
        w_energy = _get_weight_array(out, weight_for_energy, absolute=True)

        old_mean_ke = _weighted_mean_kinetic_energy_eV_from_momentum(
            px,
            py,
            pz,
            w_energy,
            m0=m0,
            q=q,
            c=c,
        )

        def objective(delta: float) -> float:
            pz_trial = _build_full_pz(delta)
            new_mean_ke = _weighted_mean_kinetic_energy_eV_from_momentum(
                px,
                py,
                pz_trial,
                w_energy,
                m0=m0,
                q=q,
                c=c,
            )
            return new_mean_ke - old_mean_ke

        f0 = objective(0.0)
        if abs(f0) > 1e-16:
            # Build an initial bracketing scale from the current longitudinal momentum spread
            pz_scale = float(np.std(pz[m]))
            line_scale = float(abs(slope) * (np.max(z[m]) - np.min(z[m])))
            scale = max(pz_scale, line_scale, 1.0)

            a = -scale
            b = +scale
            fa = objective(a)
            fb = objective(b)

            n_expand = 0
            while fa * fb > 0.0 and n_expand < 50:
                scale *= 2.0
                a = -scale
                b = +scale
                fa = objective(a)
                fb = objective(b)
                n_expand += 1

            if fa * fb > 0.0:
                raise RuntimeError(
                    "Failed to bracket the offset needed to preserve mean kinetic energy."
                )

            delta_offset = float(brentq(objective, a, b))

    pz_new = _build_full_pz(delta_offset)
    
    return _replace_velocity_from_momentum(
        out,
        px,
        py,
        pz_new,
        m0=m0,
        q=q,
        c=c,
        inplace=True,
    )



def linearize_pz_trend(
    dist: "ParticleDistribution",
    *,
    z_key: str = "z",
    pz_key: str = "pz",
    fit_weight: Union[None, str, ArrayLike] = None,
    fit_mask: Optional[Union[np.ndarray, Sequence[bool]]] = None,
    fit_bins: Optional[int] = None,
    use_binned_linear_fit: bool = False,
    x_range: Optional[tuple[float, float]] = None,
    stat: Literal["mean", "weighted_mean", "median"] = "weighted_mean",
    min_count: int = 1,
    min_weight_sum: float = 0.0,
    residual_mode: ResidualMode = "preserve",
    residual_scale: float = 1.0,
    apply_mask: Optional[Union[np.ndarray, Sequence[bool]]] = None,
    m0: float = g_m0,
    q: float = g_e0,
    c: float = g_c,
    inplace: bool = False,
) -> "ParticleDistribution":
    """
    Replace the current pz(z) trend by its best-fit linear trend.

    This is the direct answer to the earlier chirp issue:
    - residual_mode="preserve"
        preserves nonlinear residuals around the original fitted trend,
        so the final distribution is not strictly linear particle-by-particle.
    - residual_mode="discard"
        forces the final pz(z) relation to lie exactly on the fitted line
        for the selected particles.
    - residual_mode="rescale"
        keeps the same residual structure but scales its amplitude.

    Parameters
    ----------
    fit_weight : None, str, or array-like
        Weight used for the linear fit.
    fit_mask : array-like of bool, optional
        Mask defining which particles are used to determine the linear fit.
    fit_bins : int, optional
        Number of bins if use_binned_linear_fit=True.
    use_binned_linear_fit : bool
        If True, fit linearly to a binned profile first.
    x_range : tuple[float, float], optional
        Optional z range used when determining the fitted line.
    apply_mask : array-like of bool, optional
        Mask defining which particles are modified. If None, all particles are modified.
    """
    linear_fit = fit_linear_chirp(
        dist,
        z_key=z_key,
        pz_key=pz_key,
        weight=fit_weight,
        mask=fit_mask,
        bins=fit_bins,
        x_range=x_range,
        use_binned_profile=use_binned_linear_fit,
        stat=stat,
        min_count=min_count,
        min_weight_sum=min_weight_sum,
    )

    return set_pz_trend(
        dist,
        linear_fit,
        z_key=z_key,
        pz_key=pz_key,
        original_trend=None,
        fit_original_if_missing=True,
        fit_bins=100 if fit_bins is None else fit_bins,
        fit_weight=fit_weight,
        fit_stat=stat,
        fit_min_count=min_count,
        fit_min_weight_sum=min_weight_sum,
        fit_method="spline",
        fit_poly_order=1,
        fit_spline_smoothing=0.0,
        fit_spline_weight="sqrt_count",
        residual_mode=residual_mode,
        residual_scale=residual_scale,
        mask=apply_mask,
        m0=m0,
        q=q,
        c=c,
        inplace=inplace,
    )


def add_linear_chirp(
    dist: "ParticleDistribution",
    slope: float,
    *,
    intercept: float = 0.0,
    z_key: str = "z",
    pz_key: str = "pz",
    center_x: bool = False,
    weight_for_centroid: Union[None, str, ArrayLike] = None,
    mask: Optional[Union[np.ndarray, Sequence[bool]]] = None,
    m0: float = g_m0,
    q: float = g_e0,
    c: float = g_c,
    inplace: bool = False,
) -> "ParticleDistribution":
    """
    Add a linear delta chirp to pz:
        delta_pz(z) = slope * z + intercept

    If center_x=True, use:
        delta_pz(z) = slope * (z - <z>) + intercept
    """
    n = len(dist)
    z = _extract_data(dist, z_key, n_expected=n, dtype=float, name=z_key)
    w = _get_weight_array(dist, weight_for_centroid, absolute=True)
    m = _normalize_mask(mask, n)

    if center_x:
        if np.sum(w[m]) <= 0.0:
            raise ValueError("Centering requires strictly positive total selected weight.")
        z_ref = float(np.sum(w[m] * z[m]) / np.sum(w[m]))
    else:
        z_ref = 0.0

    def delta(z_in: ArrayLike) -> np.ndarray:
        z_arr = np.asarray(z_in, dtype=float)
        return slope * (z_arr - z_ref) + float(intercept)

    return apply_pz_delta(
        dist,
        delta,
        z_key=z_key,
        pz_key=pz_key,
        mask=mask,
        m0=m0,
        q=q,
        c=c,
        inplace=inplace,
    )

def match_twiss_plane(
    dist,
    *,
    plane: Literal["x", "y"],
    alpha: float,
    beta: float,
    weight: Union[None, str, ArrayLike] = None,
    mask: Optional[Union[np.ndarray, Sequence[bool]]] = None,
    center_before_match: bool = True,
    preserve_centroid: bool = True,
    inplace: bool = False,
):
    """
    Match one transverse plane to target Twiss parameters using the
    Courant-Snyder transformation, working directly in momentum slope space.

    Definitions
    -----------
    - x plane: u = x,  u' = px / pz
    - y plane: u = y,  u' = py / pz

    Parameters
    ----------
    center_before_match : bool
        If True, use centered covariance to compute and apply the match.
        If False, use raw second moments about zero, matching the standalone
        reference script convention more closely.
    preserve_centroid : bool
        Only used when center_before_match=True.
    """
    if plane not in {"x", "y"}:
        raise ValueError("plane must be 'x' or 'y'.")
    if beta <= 0.0:
        raise ValueError("Target beta must be positive.")

    out = _copy_or_inplace(dist, inplace=inplace)
    n = len(out)

    w = _get_weight_array(out, weight, absolute=True)
    m = _normalize_mask(mask, n)

    if plane == "x":
        u = _extract_data(out, "x", n_expected=n, dtype=float, name="x")
        pp = _extract_data(out, "px", n_expected=n, dtype=float, name="px")
    else:
        u = _extract_data(out, "y", n_expected=n, dtype=float, name="y")
        pp = _extract_data(out, "py", n_expected=n, dtype=float, name="py")

    pz = _extract_data(out, "pz", n_expected=n, dtype=float, name="pz")

    valid = (
        m
        & np.isfinite(u)
        & np.isfinite(pp)
        & np.isfinite(pz)
        & np.isfinite(w)
        & (np.abs(pz) > 0.0)
    )

    if np.count_nonzero(valid) < 2:
        raise ValueError("At least two valid selected particles are required for Twiss matching.")
    if float(np.sum(w[valid])) <= 0.0:
        raise ValueError("Selected particles must have strictly positive total weight.")

    up = pp / pz

    u_sel = u[valid]
    up_sel = up[valid]
    pz_sel = pz[valid]
    w_sel = w[valid]

    if center_before_match:
        alpha_old, beta_old, _eps_old, mean_u, mean_up = _weighted_centered_twiss_from_arrays(
            u_sel, up_sel, w_sel
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

    # Courant-Snyder transformation matrix
    r11 = np.sqrt(beta / beta_old)
    r12 = 0.0
    r21 = (alpha_old - alpha) / np.sqrt(beta_old * beta)
    r22 = np.sqrt(beta_old / beta)

    work_u_new = r11 * work_u + r12 * work_up
    work_up_new = r21 * work_u + r22 * work_up

    if center_before_match:
        if preserve_centroid:
            u_new_sel = work_u_new + mean_u
            up_new_sel = work_up_new + mean_up
        else:
            u_new_sel = work_u_new
            up_new_sel = work_up_new
    else:
        u_new_sel = work_u_new
        up_new_sel = work_up_new

    u_new  = u.copy()
    pp_new = pp.copy()

    u_new[valid]  = u_new_sel
    pp_new[valid] = up_new_sel * pz_sel

    if plane == "x":
        out.update_quantity("x", u_new)
        return _update_momentum_components(out, pp_new, out.py, out.pz, inplace=True)

    out.update_quantity("y", u_new)
    return _update_momentum_components(out, out.px, pp_new, out.pz, inplace=True)


def match_twiss_x(
    dist,
    alpha: float,
    beta: float,
    *,
    weight: Union[None, str, ArrayLike] = None,
    mask: Optional[Union[np.ndarray, Sequence[bool]]] = None,
    center_before_match: bool = True,
    preserve_centroid: bool = True,
    inplace: bool = False,
):
    """
    Match Twiss parameters in the x plane.
    """
    return match_twiss_plane(
        dist,
        plane="x",
        alpha=alpha,
        beta=beta,
        weight=weight,
        mask=mask,
        center_before_match=center_before_match,
        preserve_centroid=preserve_centroid,
        inplace=inplace,
    )


def match_twiss_y(
    dist,
    alpha: float,
    beta: float,
    *,
    weight: Union[None, str, ArrayLike] = None,
    mask: Optional[Union[np.ndarray, Sequence[bool]]] = None,
    center_before_match: bool = True,
    preserve_centroid: bool = True,
    inplace: bool = False,
):
    """
    Match Twiss parameters in the y plane.
    """
    return match_twiss_plane(
        dist,
        plane="y",
        alpha=alpha,
        beta=beta,
        weight=weight,
        mask=mask,
        center_before_match=center_before_match,
        preserve_centroid=preserve_centroid,
        inplace=inplace,
    )

def match_twiss_xy(
    dist,
    alpha_x: float,
    beta_x: float,
    alpha_y: float,
    beta_y: float,
    *,
    weight: Union[None, str, ArrayLike] = 'Q',
    mask: Optional[Union[np.ndarray, Sequence[bool]]] = None,
    center_before_match: bool = True,
    preserve_centroid: bool = True,
    inplace: bool = False,
):
    """
    Match both x and y transverse planes sequentially.
    
    This is a convenience wrapper around `match_twiss_x(...)` and
    `match_twiss_y(...)`. The x-plane matching is applied first, followed by
    the y-plane matching.
    
    Parameters
    ----------
    dist : ParticleDistribution
        Input distribution.
    alpha_x : float
        Target Twiss alpha in the x plane.
    beta_x : float
        Target Twiss beta [m] in the x plane.
    alpha_y : float
        Target Twiss alpha in the y plane.
    beta_y : float
        Target Twiss beta [m] in the y plane.
    weight : None, str, or array-like
        Optional particle weights used when computing the current Twiss
        parameters.
    mask : array-like of bool, optional
        Particle mask selecting which particles are transformed.
    center_before_match : bool
        If True, compute and match Twiss parameters using centered second
        moments (covariance-based definition). If False, use raw second
        moments about zero.
    preserve_centroid : bool
        Only meaningful when `center_before_match=True`.
        If True, preserve the weighted centroid in each matched plane.
        If False, center the selected particles at zero after matching.
    inplace : bool
        Whether to modify the input distribution directly.
    
    Returns
    -------
    ParticleDistribution
        Updated distribution with both transverse planes matched.
    
    Notes
    -----
    This function is intentionally designed as a *transverse-only* manipulator.
    
    It changes the transverse phase-space coordinates in x and y, while leaving
    the longitudinal component unchanged in the current implementation.
    As a consequence, the beam mean energy is generally **not** preserved.
    
    This behavior is intentional: preserving the beam energy would require
    modifying the longitudinal momentum/velocity as well, which would couple
    the transverse matching to the longitudinal phase space and could change
    the chirp. The current design avoids that coupling.
    
    Therefore:
    - transverse Twiss matching is kept independent of longitudinal
      manipulations,
    - a shift in mean energy after matching is expected in general,
    - and this should not be interpreted as a bug.
    
    If exact energy preservation is needed in the future, it should be offered
    as a separate optional mode, since it would necessarily alter the
    longitudinal dynamics.
    """
    out = _copy_or_inplace(dist, inplace=inplace)

    out = match_twiss_x(
        out,
        alpha=alpha_x,
        beta=beta_x,
        weight=weight,
        mask=mask,
        center_before_match=center_before_match,
        preserve_centroid=preserve_centroid,
        inplace=True,
    )

    out = match_twiss_y(
        out,
        alpha=alpha_y,
        beta=beta_y,
        weight=weight,
        mask=mask,
        center_before_match=center_before_match,
        preserve_centroid=preserve_centroid,
        inplace=True,
    )

    return out


def replicate_longitudinally(
    dist: ParticleDistribution,
    n_bunches: int,
    spacing: float,
    *,
    spacing_mode: Literal["z", "t", "co_moving"] = "z",
    charge_mode: Literal["preserve_per_bunch", "preserve_total"] = "preserve_per_bunch",
    reference_velocity: Optional[float] = None,
    add_copy_index: bool = True,
    sort_by: Optional[Literal["z", "t"]] = None,
) -> ParticleDistribution:
    """
    Replicate a particle distribution into a longitudinal train.

    Parameters
    ----------
    dist : ParticleDistribution
        Input distribution.
    n_bunches : int
        Total number of bunches in the output, including the original one.
        Must satisfy n_bunches >= 1.
    spacing : float
        Constant interval between adjacent bunches.

        Interpretation depends on spacing_mode:
        - "z": spacing is Δz [m]
        - "t": spacing is Δt [s]
        - "co_moving": spacing is interpreted as Δz [m], and Δt = Δz / v_ref
    spacing_mode : {"z", "t", "co_moving"}
        Longitudinal spacing convention.
    charge_mode : {"preserve_per_bunch", "preserve_total"}
        Charge scaling policy.
        - "preserve_per_bunch": each bunch keeps the original Q array
        - "preserve_total": total charge of the whole train equals the original total charge
    reference_velocity : float, optional
        Reference longitudinal velocity used only for spacing_mode="co_moving".
        If omitted, it is inferred from the distribution.
    add_copy_index : bool
        If True, add an integer extra quantity named "copy_index".
    sort_by : {"z", "t", None}
        If not None, sort the output distribution by the chosen longitudinal quantity.

    Returns
    -------
    ParticleDistribution
        Replicated distribution.

    Notes
    -----
    This function replicates:
    - all base quantities
    - all existing extra quantities

    Derived quantities are not explicitly copied; they remain derived in the
    returned ParticleDistribution.
    """
    if n_bunches < 1:
        raise ValueError(f"n_bunches must be >= 1, got {n_bunches}.")
    if spacing_mode not in {"z", "t", "co_moving"}:
        raise ValueError(
            f"spacing_mode must be 'z', 't', or 'co_moving', got {spacing_mode!r}."
        )
    if charge_mode not in {"preserve_per_bunch", "preserve_total"}:
        raise ValueError(
            "charge_mode must be 'preserve_per_bunch' or 'preserve_total', "
            f"got {charge_mode!r}."
        )
    if sort_by not in {None, "z", "t"}:
        raise ValueError(f"sort_by must be None, 'z', or 't', got {sort_by!r}.")

    if spacing_mode == "co_moving":
        v_ref = _infer_reference_velocity(dist) if reference_velocity is None else float(reference_velocity)
        if v_ref == 0.0:
            raise ValueError("reference_velocity must be nonzero for spacing_mode='co_moving'.")
    else:
        v_ref = None

    base_keys = ("x", "y", "z", "px", "py", "pz", "t", "Q")
    extra_keys = [k for k in dist.quantity_keys if dist.quantity_kind(k) == "extra"]

    base_data = {k: np.asarray(dist.get_data(k)).reshape(-1) for k in base_keys}
    extra_quantities = {k: dist.get_quantity(k) for k in extra_keys}

    n_particles = len(base_data["x"])
    if any(len(base_data[k]) != n_particles for k in base_keys):
        raise ValueError("Base quantity lengths are inconsistent.")

    # Prepare list-of-blocks for concatenation
    base_blocks = {k: [] for k in base_keys}
    extra_blocks = {k: [] for k in extra_keys}
    copy_index_blocks = []

    q_scale = 1.0 if charge_mode == "preserve_per_bunch" else 1.0 / n_bunches

    for i in range(n_bunches):
        z_shift = 0.0
        t_shift = 0.0

        if spacing_mode == "z":
            z_shift = i * spacing
        elif spacing_mode == "t":
            t_shift = i * spacing
        elif spacing_mode == "co_moving":
            z_shift = i * spacing
            t_shift = z_shift / v_ref

        for k in base_keys:
            arr = base_data[k].copy()

            if k == "z":
                arr = arr + z_shift
            elif k == "t":
                arr = arr + t_shift
            elif k == "Q":
                arr = arr * q_scale

            base_blocks[k].append(arr)

        for k in extra_keys:
            arr = np.asarray(extra_quantities[k].data).reshape(-1).copy()
            extra_blocks[k].append(arr)

        if add_copy_index:
            copy_index_blocks.append(np.full(n_particles, i, dtype=int))

    # Concatenate all blocks
    base_concat = {k: np.concatenate(v, axis=0) for k, v in base_blocks.items()}
    extra_concat = {k: np.concatenate(v, axis=0) for k, v in extra_blocks.items()}
    copy_index_concat = (
        np.concatenate(copy_index_blocks, axis=0) if add_copy_index else None
    )

    # Optional sorting
    if sort_by is not None:
        order = np.argsort(base_concat[sort_by], kind="stable")

        for k in base_concat:
            base_concat[k] = base_concat[k][order]

        for k in extra_concat:
            extra_concat[k] = extra_concat[k][order]

        if copy_index_concat is not None:
            copy_index_concat = copy_index_concat[order]

    # Rebuild the base distribution
    out = ParticleDistribution.from_arrays(
        x=base_concat["x"],
        y=base_concat["y"],
        z=base_concat["z"],
        px=base_concat["px"],
        py=base_concat["py"],
        pz=base_concat["pz"],
        t=base_concat["t"],
        Q=base_concat["Q"],
    )

    # Rebuild extras with original metadata
    for k, q in extra_quantities.items():
        out._quantities[k] = ParticleArrayQuantity(
            name=q.name,
            data=extra_concat[k],
            unit=q.unit,
            dtype_kind=q.dtype_kind,
            short_name=q.short_name,
            long_name=q.long_name,
            latex_name=q.latex_name,
            category=q.category,
            is_derived=False,
            is_discrete=q.is_discrete,
            preferred_scale=q.preferred_scale,
        )

    if add_copy_index:
        out._quantities["copy_index"] = ParticleArrayQuantity(
            name="copy_index",
            data=copy_index_concat,
            unit="",
            dtype_kind="int",
            short_name="copy_index",
            long_name="copy index",
            latex_name="i_{copy}",
            category="flag",
            is_derived=False,
            is_discrete=True,
            preferred_scale=1.0,
        )

    return out

def multiply_longitudinal_profile(
    dist,
    curve: CurveLike,
    *,
    coordinate: str = "z",
    center: Optional[Union[str, float]] = None,
    normalize: bool = False,
    allow_negative: bool = False,
    outside_value: float = 0.0,
    inplace: bool = False,
):
    """
    Multiply the longitudinal charge/current profile by a curve.

    This function modifies the particle macro-charges according to:
        Q_new = Q_old * f(xi)

    where xi is the chosen longitudinal coordinate (typically z or t),
    optionally shifted by a center value.

    Parameters
    ----------
    dist : ParticleDistribution
        Input distribution.
    curve : callable or (x_sample, y_sample)
        Multiplicative curve.
        - callable: f(x_eval)
        - tuple: sampled curve (x_sample, y_sample), internally interpolated
    coordinate : str
        Longitudinal coordinate key used to evaluate the curve, typically "z" or "t".
    center : None, "mean", "centroid", or float
        Optional coordinate shift before evaluating the curve.
        - None: use raw coordinate
        - "mean": subtract the beam centroid of the chosen coordinate
        - "centroid": same as "mean"
        - float: subtract this value
    normalize : bool
        If True, rescale the resulting Q array so that the total signed charge
        is preserved.
    allow_negative : bool
        If False, negative curve values are not allowed.
    outside_value : float
        Value used outside the sampled range when curve is given as arrays.
    inplace : bool
        Whether to modify the input distribution directly.

    Returns
    -------
    ParticleDistribution
        Updated distribution.

    Notes
    -----
    - This is a charge-weighting operation, not a resampling operation.
    - If all particles have the same sign of Q, preserving total signed charge
      is equivalent to preserving total charge magnitude.
    """
    out = _copy_or_inplace(dist, inplace=inplace)
    n = len(out)

    xi = _extract_data(out, coordinate, n_expected=n, dtype=float, name=coordinate)
    q_old = _extract_data(out, "Q", n_expected=n, dtype=float, name="Q")
        
    if center is None:
        xi_eval = xi
    elif isinstance(center, str):
        if center not in {"mean", "centroid"}:
            raise ValueError("center must be None, 'mean', 'centroid', or a float.")
        xi_eval = xi - float(out.centroid()[coordinate])
    else:
        xi_eval = xi - float(center)

    factors = _evaluate_curve_like(
        xi_eval,
        curve,
        outside_value=outside_value,
    )

    if not allow_negative and np.any(factors < 0.0):
        factors = np.maximum(factors, 0)
        # raise ValueError(
        #     "The evaluated curve contains negative values, but allow_negative=False."
        # )

    q_new = q_old * factors

    if normalize:
        total_old = float(np.sum(q_old))
        total_new = float(np.sum(q_new))

        if total_new == 0.0:
            raise ValueError(
                "Cannot normalize because the new total signed charge is zero."
            )

        q_new = q_new * (total_old / total_new)

    out.update_quantity("Q", q_new)
    return out


def center_beam(
    dist: "ParticleDistribution",
    *,
    x_key: str = "x",
    y_key: str = "y",
    z_key: str = "z",
    weight: Union[None, str, ArrayLike] = "Q_abs",
    inplace: bool = False,
) -> "ParticleDistribution":
    """
    Shift the beam so that its charge-weighted centroid is at (0, 0, 0).

    Each spatial coordinate is shifted by subtracting its charge-weighted mean.

    Parameters
    ----------
    dist : ParticleDistribution
        Input distribution.
    x_key, y_key, z_key : str
        Quantity keys for the three spatial coordinates.
    weight : None, str, or array-like
        Particle weights. Defaults to 'Q_abs'.
    inplace : bool
        Whether to modify the input distribution directly.

    Returns
    -------
    ParticleDistribution
        Distribution with charge-weighted centroid shifted to the origin.
    """
    out = _copy_or_inplace(dist, inplace=inplace)
    n = len(out)
    w = _get_weight_array(out, weight, absolute=True)
    for key in (x_key, y_key, z_key):
        arr = _extract_data(out, key, n_expected=n, dtype=float, name=key)
        out.update_quantity(key, arr - _weighted_mean_1d(arr, w))
    return out


# ---------------------------------------------------------------------------
# Transport matrix transformations
# ---------------------------------------------------------------------------

def _apply_transport_matrix_core(
    out: "ParticleDistribution",
    M6: np.ndarray,
    w: np.ndarray,
) -> "ParticleDistribution":
    """
    Apply a 6×6 transport matrix to the state-vector deviations.

    The state vector is u = (x, x', y, y', z, δ) where:
        x' = px / |p|,  y' = py / |p|
        δ  = (|p| - p_ref) / p_ref,  p_ref = weighted mean of |p|

    The matrix acts on per-particle deviations from the weighted centroid::

        Δu_new = M6 @ Δu,    u_new = u_centroid + Δu_new

    After the transformation the new momenta are reconstructed as::

        |p|_new = p_ref * (1 + δ_new)
        px_new  = |p|_new * x'_new
        py_new  = |p|_new * y'_new
        pz_new  = sqrt(|p|_new² - px_new² - py_new²)  [sign preserved]

    Modifies `out` in-place; the caller is responsible for copying first.
    """
    p_abs = out.p_abs
    p_ref = float(np.average(p_abs, weights=w))

    px = out._quantities["px"].data.copy()
    py = out._quantities["py"].data.copy()
    pz = out._quantities["pz"].data.copy()
    x  = out._quantities["x"].data.copy()
    y  = out._quantities["y"].data.copy()
    z  = out._quantities["z"].data.copy()

    xp    = px / p_abs
    yp    = py / p_abs
    delta = (p_abs - p_ref) / p_ref

    state = np.stack([x, xp, y, yp, z, delta])          # (6, n)
    centroid = np.average(state, weights=w, axis=1, keepdims=True)  # (6, 1)
    state_new = centroid + M6 @ (state - centroid)

    x_new, xp_new, y_new, yp_new, z_new, delta_new = state_new

    p_abs_new = p_ref * (1.0 + delta_new)
    px_new    = p_abs_new * xp_new
    py_new    = p_abs_new * yp_new
    pz_sq     = p_abs_new**2 - px_new**2 - py_new**2

    if np.any(pz_sq < 0.0):
        raise ValueError(
            "Transport matrix produces an unphysical state: transverse momentum "
            "exceeds total momentum for at least one particle."
        )

    pz_new = np.copysign(np.sqrt(pz_sq), pz)

    out.update_quantity("x",  x_new)
    out.update_quantity("y",  y_new)
    out.update_quantity("z",  z_new)
    out.update_quantity("px", px_new)
    out.update_quantity("py", py_new)
    out.update_quantity("pz", pz_new)
    return out


def apply_matrix_x(
    dist: "ParticleDistribution",
    M: np.ndarray,
    *,
    weight: Union[None, str, ArrayLike] = "Q_abs",
    inplace: bool = False,
) -> "ParticleDistribution":
    """
    Apply a 2×2 transport matrix to the horizontal phase space (x, x').

    The matrix acts on deviations from the charge-weighted centroid.
    y, y', z, and δ are left unchanged.

    State vector convention and momentum reconstruction are described in
    :func:`_apply_transport_matrix_core`.

    Parameters
    ----------
    dist : ParticleDistribution
        Input distribution.
    M : array-like, shape (2, 2)
        Transport matrix acting on (x, x').
    weight : None, str, or array-like
        Particle weights used for centroid and p_ref. Defaults to 'Q_abs'.
    inplace : bool
        Whether to modify the input distribution directly.

    Returns
    -------
    ParticleDistribution
        Transformed distribution.
    """
    M = np.asarray(M, dtype=float)
    if M.shape != (2, 2):
        raise ValueError(f"M must be a 2×2 matrix, got shape {M.shape}.")
    M6 = np.eye(6)
    M6[np.ix_([0, 1], [0, 1])] = M
    out = _copy_or_inplace(dist, inplace=inplace)
    return _apply_transport_matrix_core(out, M6, _get_weight_array(out, weight, absolute=True))


def apply_matrix_y(
    dist: "ParticleDistribution",
    M: np.ndarray,
    *,
    weight: Union[None, str, ArrayLike] = "Q_abs",
    inplace: bool = False,
) -> "ParticleDistribution":
    """
    Apply a 2×2 transport matrix to the vertical phase space (y, y').

    The matrix acts on deviations from the charge-weighted centroid.
    x, x', z, and δ are left unchanged.

    State vector convention and momentum reconstruction are described in
    :func:`_apply_transport_matrix_core`.

    Parameters
    ----------
    dist : ParticleDistribution
        Input distribution.
    M : array-like, shape (2, 2)
        Transport matrix acting on (y, y').
    weight : None, str, or array-like
        Particle weights used for centroid and p_ref. Defaults to 'Q_abs'.
    inplace : bool
        Whether to modify the input distribution directly.

    Returns
    -------
    ParticleDistribution
        Transformed distribution.
    """
    M = np.asarray(M, dtype=float)
    if M.shape != (2, 2):
        raise ValueError(f"M must be a 2×2 matrix, got shape {M.shape}.")
    M6 = np.eye(6)
    M6[np.ix_([2, 3], [2, 3])] = M
    out = _copy_or_inplace(dist, inplace=inplace)
    return _apply_transport_matrix_core(out, M6, _get_weight_array(out, weight, absolute=True))


def apply_matrix_z(
    dist: "ParticleDistribution",
    M: np.ndarray,
    *,
    weight: Union[None, str, ArrayLike] = "Q_abs",
    convention: Literal["z", "tau"] = "z",
    inplace: bool = False,
) -> "ParticleDistribution":
    """
    Apply a 2×2 transport matrix to the longitudinal phase space (z, δ).

    The matrix acts on deviations from the charge-weighted centroid.
    x, x', y, and y' are left unchanged.

    State vector convention and momentum reconstruction are described in
    :func:`_apply_transport_matrix_core`.

    Parameters
    ----------
    dist : ParticleDistribution
        Input distribution.
    M : array-like, shape (2, 2)
        Transport matrix acting on (z, δ) or (τ, δ) depending on `convention`.
    weight : None, str, or array-like
        Particle weights used for centroid and p_ref. Defaults to 'Q_abs'.
    convention : {"z", "tau"}
        Longitudinal coordinate convention of the input matrix M.

        - ``"z"``  : M acts on (z, δ) where z > 0 means ahead.
          Compressing chicane: M[0,1] = R56 < 0.
        - ``"tau"``: M acts on (τ, δ) where τ = z_ref − z, τ > 0 means lagging.
          Compressing chicane: M[0,1] = R56 > 0 (standard beam-optics textbook sign).
          The matrix is converted to z convention internally via S·M·S,
          where S = diag(−1, 1).
    inplace : bool
        Whether to modify the input distribution directly.

    Returns
    -------
    ParticleDistribution
        Transformed distribution.
    """
    M = np.asarray(M, dtype=float)
    if M.shape != (2, 2):
        raise ValueError(f"M must be a 2×2 matrix, got shape {M.shape}.")
    if convention == "tau":
        # S = diag(-1, 1);  M_z = S @ M_tau @ S  flips off-diagonal signs
        M = M * np.array([[1., -1.], [-1., 1.]])
    elif convention != "z":
        raise ValueError(f"convention must be 'z' or 'tau', got {convention!r}.")
    M6 = np.eye(6)
    M6[np.ix_([4, 5], [4, 5])] = M
    out = _copy_or_inplace(dist, inplace=inplace)
    return _apply_transport_matrix_core(out, M6, _get_weight_array(out, weight, absolute=True))


def apply_chicane_map(
    dist: "ParticleDistribution",
    R56: float,
    *,
    T566: float = 0.0,
    U5666: float = 0.0,
    weight: Union[None, str, ArrayLike] = "Q_abs",
    convention: Literal["z", "tau"] = "z",
    inplace: bool = False,
) -> "ParticleDistribution":
    """
    Apply a chicane longitudinal transfer map up to third order.

    A magnetic chicane conserves particle momentum; only the path length (and
    hence z) changes.  The map is::

        z_new = z + R56·δ + T566·δ² + U5666·δ³
        δ_new = δ         (momenta unchanged)

    where ``δ = (|p| − p_ref) / p_ref`` is the relative momentum deviation and
    ``p_ref`` is the charge-weighted mean total momentum.

    Relation to ``apply_matrix_z``
    --------------------------------
    ``apply_matrix_z`` with the standard chicane matrix (z convention)::

        M = [[1, R56],
             [0, 1  ]]

    is equivalent to calling this function with ``T566=0`` and ``U5666=0``.
    Use this function when second- or third-order chromatic effects matter.

    Typical parameter values (four-dipole symmetric chicane)
    --------------------------------------------------------
    For a four-dipole chicane with half-bend angle θ and bend radius ρ::

        R56  ≈  −2θ²L_d   (L_d = drift length between dipoles)
        T566 ≈  1.5 × |R56|
        U5666 ≈ ...        (usually negligible)

    Parameters
    ----------
    dist : ParticleDistribution
        Input distribution.
    R56 : float
        First-order momentum compaction [m].

        Interpretation depends on ``convention``:

        - ``"z"``  : R56 < 0 for a compressing chicane (δ > 0 at the bunch head).
        - ``"tau"``: R56 > 0 for a compressing chicane (standard textbook sign,
          where R56 = ∂τ/∂δ and τ > 0 means lagging).
    T566 : float
        Second-order term [m].  Sign follows the same convention as R56.  Default 0.
    U5666 : float
        Third-order term [m].  Sign follows the same convention as R56.  Default 0.
    weight : None, str, or array-like
        Particle weights used to define ``p_ref``.  Defaults to ``'Q_abs'``.
    convention : {"z", "tau"}
        Sign convention for R56, T566, U5666.  Default ``"z"``.
    inplace : bool
        Whether to modify the input distribution directly.

    Returns
    -------
    ParticleDistribution
        Distribution after applying the chicane map.
    """
    if convention == "tau":
        R56, T566, U5666 = -R56, -T566, -U5666
    elif convention != "z":
        raise ValueError(f"convention must be 'z' or 'tau', got {convention!r}.")

    out = _copy_or_inplace(dist, inplace=inplace)
    w = _get_weight_array(out, weight, absolute=True)

    p_abs = out.p_abs                                   # (n,)  [eV/c]
    p_ref = float(np.average(p_abs, weights=w))
    delta = (p_abs - p_ref) / p_ref                     # (n,)  dimensionless

    z = out._quantities["z"].data.copy()
    dz = float(R56) * delta
    if T566 != 0.0:
        dz = dz + float(T566) * delta**2
    if U5666 != 0.0:
        dz = dz + float(U5666) * delta**3

    out.update_quantity("z", z + dz)
    return out


def apply_matrix_xy(
    dist: "ParticleDistribution",
    M: np.ndarray,
    *,
    weight: Union[None, str, ArrayLike] = "Q_abs",
    inplace: bool = False,
) -> "ParticleDistribution":
    """
    Apply a 4×4 transport matrix to the transverse phase space (x, x', y, y').

    The matrix acts on deviations from the charge-weighted centroid.
    z and δ are left unchanged.

    State vector convention and momentum reconstruction are described in
    :func:`_apply_transport_matrix_core`.

    Parameters
    ----------
    dist : ParticleDistribution
        Input distribution.
    M : array-like, shape (4, 4)
        Transport matrix acting on (x, x', y, y').
    weight : None, str, or array-like
        Particle weights used for centroid and p_ref. Defaults to 'Q_abs'.
    inplace : bool
        Whether to modify the input distribution directly.

    Returns
    -------
    ParticleDistribution
        Transformed distribution.
    """
    M = np.asarray(M, dtype=float)
    if M.shape != (4, 4):
        raise ValueError(f"M must be a 4×4 matrix, got shape {M.shape}.")
    M6 = np.eye(6)
    M6[np.ix_([0, 1, 2, 3], [0, 1, 2, 3])] = M
    out = _copy_or_inplace(dist, inplace=inplace)
    return _apply_transport_matrix_core(out, M6, _get_weight_array(out, weight, absolute=True))


_S6 = np.diag([1., 1., 1., 1., -1., 1.])


def apply_matrix_6d(
    dist: "ParticleDistribution",
    M: np.ndarray,
    *,
    weight: Union[None, str, ArrayLike] = "Q_abs",
    convention: Literal["z", "tau"] = "z",
    inplace: bool = False,
) -> "ParticleDistribution":
    """
    Apply a 6×6 transport matrix to the full phase space (x, x', y, y', z, δ).

    The matrix acts on deviations from the charge-weighted centroid::

        Δu_new = M @ Δu,    u_new = u_centroid + Δu_new

    where u = (x, x', y, y', z, δ) and:
        x' = px / |p|,  y' = py / |p|
        δ  = (|p| - p_ref) / p_ref,  p_ref = weighted mean of |p|

    After the transformation the momenta are reconstructed as::

        |p|_new = p_ref * (1 + δ_new)
        px_new  = |p|_new * x'_new
        py_new  = |p|_new * y'_new
        pz_new  = sqrt(|p|_new² - px_new² - py_new²)  [sign of pz preserved]

    Parameters
    ----------
    dist : ParticleDistribution
        Input distribution.
    M : array-like, shape (6, 6)
        Transport matrix acting on (x, x', y, y', z, δ) or (x, x', y, y', τ, δ)
        depending on `convention`.
    weight : None, str, or array-like
        Particle weights used for centroid and p_ref. Defaults to 'Q_abs'.
    convention : {"z", "tau"}
        Longitudinal coordinate convention of the input matrix M.

        - ``"z"``  : M acts on (x, x', y, y', z, δ), z > 0 ahead.
          Compressing chicane: M[4,5] = R56 < 0.
        - ``"tau"``: M acts on (x, x', y, y', τ, δ), τ > 0 lagging (standard textbook).
          Compressing chicane: M[4,5] = R56 > 0.
          Converted to z convention internally via S·M·S,
          where S = diag(1, 1, 1, 1, −1, 1).
    inplace : bool
        Whether to modify the input distribution directly.

    Returns
    -------
    ParticleDistribution
        Transformed distribution.
    """
    M = np.asarray(M, dtype=float)
    if M.shape != (6, 6):
        raise ValueError(f"M must be a 6×6 matrix, got shape {M.shape}.")
    if convention == "tau":
        M = _S6 @ M @ _S6
    elif convention != "z":
        raise ValueError(f"convention must be 'z' or 'tau', got {convention!r}.")
    out = _copy_or_inplace(dist, inplace=inplace)
    return _apply_transport_matrix_core(out, M, _get_weight_array(out, weight, absolute=True))






















