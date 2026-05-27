"""
Diagnostics for ParticleDistribution3D and SliceDistribution.

Twiss/emittance, binned profiles and trend fits, longitudinal linearity,
beam diagnostics, current-profile fits. All results returned as frozen
dataclasses; see PhaseSpacePlaneResult, BeamDiagnosticsResult, etc.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Callable, Dict, Literal, Optional, Sequence, Union, TYPE_CHECKING

import numpy as np
from scipy.interpolate import UnivariateSpline

from scipy.constants import c as g_c, m_e as g_m0
from .utils import _extract_data, _get_weight_array, _normalize_mask

if TYPE_CHECKING:
    from .core import ParticleDistribution


ArrayLike = Union[np.ndarray, Sequence[float], float]
StatMethod = Literal["mean", "weighted_mean", "median"]
FitMethod = Literal["linear", "poly", "spline"]

@dataclass(frozen=True)
class PhaseSpacePlaneResult:
    """
    Weighted transverse phase-space second-moment analysis result for one plane.

    Twiss is defined in (u, u') space, where:
        - for plane='x': u = x,  u' = px / pz
        - for plane='y': u = y,  u' = py / pz

    Notes
    -----
    - geometric_emittance is computed from the 2x2 covariance matrix in (u, u')
    - normalized_emittance is defined here as:
          eps_n = beta0*gamma0 * eps_geom
      where beta0*gamma0 = p_ref / (m0 c) and p_ref is the charge-weighted
      mean of |p| over the selected particles (textbook convention;
      matches MAD-X, ASTRA, Ocelot)
    """
    plane: str

    mean_u: float
    mean_up: float

    sigma_u: float
    sigma_up: float
    cov_u_up: float
    covariance: np.ndarray

    geometric_emittance: float
    normalized_emittance: float
    mean_beta_gamma: float

    alpha: float
    beta: float
    gamma_twiss: float

    n_selected: int
    weight_sum: float
    
@dataclass(frozen=True)
class BinnedProfileResult:
    """
    Result of a 1D binned profile y(x).

    Attributes
    ----------
    x_centers : np.ndarray
        Bin centers for all bins.
    y_stat : np.ndarray
        Per-bin statistic values. Invalid bins are NaN.
    y_std : np.ndarray
        Per-bin standard deviation values. Invalid bins are NaN.
    y_rms : np.ndarray
        Per-bin RMS values. Invalid bins are NaN.
    counts : np.ndarray
        Number of particles in each bin.
    weight_sum : np.ndarray
        Sum of particle weights in each bin.
    valid_mask : np.ndarray
        Boolean mask indicating valid bins.
    bin_edges : np.ndarray
        Bin edge coordinates.
    x_key : str | None
        Source quantity key if input x came from a distribution key.
    y_key : str | None
        Source quantity key if input y came from a distribution key.
    """
    x_centers: np.ndarray
    y_stat: np.ndarray
    y_std: np.ndarray
    y_rms: np.ndarray
    counts: np.ndarray
    weight_sum: np.ndarray
    valid_mask: np.ndarray
    bin_edges: np.ndarray
    x_key: Optional[str] = None
    y_key: Optional[str] = None

    @property
    def x_valid(self) -> np.ndarray:
        return self.x_centers[self.valid_mask]

    @property
    def y_valid(self) -> np.ndarray:
        return self.y_stat[self.valid_mask]

    @property
    def y_std_valid(self) -> np.ndarray:
        return self.y_std[self.valid_mask]

    @property
    def y_rms_valid(self) -> np.ndarray:
        return self.y_rms[self.valid_mask]

    @property
    def counts_valid(self) -> np.ndarray:
        return self.counts[self.valid_mask]

    @property
    def weight_sum_valid(self) -> np.ndarray:
        return self.weight_sum[self.valid_mask]


@dataclass(frozen=True)
class TrendFitResult:
    """
    Result of fitting a smooth trend y(x).

    Attributes
    ----------
    method : str
        Fit method name.
    evaluator : callable
        Callable f(x) -> y_fit.
    x_sample : np.ndarray
        x values used for fitting.
    y_sample : np.ndarray
        y values used for fitting.
    fit_params : dict
        Method-specific fit metadata.
    profile : BinnedProfileResult | None
        Optional source binned profile.
    """
    method: str
    evaluator: Callable[[ArrayLike], np.ndarray]
    x_sample: np.ndarray
    y_sample: np.ndarray
    fit_params: dict
    profile: Optional[BinnedProfileResult] = None

    def __call__(self, x: ArrayLike) -> np.ndarray:
        return np.asarray(self.evaluator(x), dtype=float)


@dataclass(frozen=True)
class LongitudinalLinearityResult:
    """
    Result of longitudinal phase-space linearity analysis.

    Attributes
    ----------
    f_nonlinear : float
        Nonlinear fraction: weighted RMS of the bin-level residuals from the
        linear chirp fit divided by the weighted std of the pz profile.
        Range [0, 1]; 0 = perfectly linear, 1 = entirely nonlinear.
    r_squared : float
        Coefficient of determination equivalent: ``1 - f_nonlinear**2``.
    profile : BinnedProfileResult
        Binned pz(z) profile used for the analysis.
    linear_trend : TrendFitResult
        Fitted linear chirp trend.
    """
    f_nonlinear: float
    r_squared: float
    profile: "BinnedProfileResult"
    linear_trend: "TrendFitResult"


@dataclass(frozen=True)
class AnalyzeLongitudinalTrendResult:
    """Bundle of pz(z) binned profile, fitted trend, and residual analysis."""
    profile: "BinnedProfileResult"
    trend: "TrendFitResult"
    residuals: "ResidualAnalysisResult"


@dataclass(frozen=True)
class ResidualAnalysisResult:
    """
    Result of residual analysis for y relative to a fitted trend f(x).

    Attributes
    ----------
    x : np.ndarray
        Input x values.
    y : np.ndarray
        Input y values.
    y_fit : np.ndarray
        Fitted trend values evaluated at x.
    residual : np.ndarray
        Residual array y - y_fit.
    weighted_mean : float
        Weighted mean of residual.
    weighted_std : float
        Weighted standard deviation of residual.
    weighted_rms : float
        Weighted RMS of residual.
    """
    x: np.ndarray
    y: np.ndarray
    y_fit: np.ndarray
    residual: np.ndarray
    weighted_mean: float
    weighted_std: float
    weighted_rms: float


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    wsum = float(np.sum(weights))
    if wsum <= 0.0:
        raise ValueError("Total weight must be positive.")
    return float(np.sum(weights * values) / wsum)


def _weighted_cov_2x2(a: np.ndarray, b: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Weighted 2x2 covariance matrix of two 1D arrays.
    """
    wsum = float(np.sum(weights))
    if wsum <= 0.0:
        raise ValueError("Total weight must be positive.")

    mean_a = _weighted_mean(a, weights)
    mean_b = _weighted_mean(b, weights)

    da = a - mean_a
    db = b - mean_b

    s11 = float(np.sum(weights * da * da) / wsum)
    s12 = float(np.sum(weights * da * db) / wsum)
    s22 = float(np.sum(weights * db * db) / wsum)

    return np.array([[s11, s12], [s12, s22]], dtype=float)


def _weighted_rms(values: np.ndarray, weights: np.ndarray) -> float:
    wsum = float(np.sum(weights))
    if wsum <= 0.0:
        raise ValueError("Total weight must be positive.")
    return float(np.sqrt(np.sum(weights * values**2) / np.sum(weights)))


def _weighted_std(values: np.ndarray, weights: np.ndarray) -> float:
    mean = _weighted_mean(values, weights)
    return float(np.sqrt(np.sum(weights * (values - mean) ** 2) / np.sum(weights)))


def _digitize_to_bins(x: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """
    Map x values to bin indices in [0, n_bins-1], or -1 for out-of-range.
    """
    idx = np.digitize(x, bin_edges) - 1
    idx[(x < bin_edges[0]) | (x > bin_edges[-1])] = -1

    # Include the right edge in the last bin.
    idx[x == bin_edges[-1]] = len(bin_edges) - 2
    return idx


def _infer_range(x: np.ndarray, x_range: Optional[tuple[float, float]]) -> tuple[float, float]:
    if x_range is not None:
        x_min, x_max = float(x_range[0]), float(x_range[1])
    else:
        x_min, x_max = float(np.min(x)), float(np.max(x))

    if not np.isfinite(x_min) or not np.isfinite(x_max):
        raise ValueError("x range contains non-finite values.")
    if x_max <= x_min:
        raise ValueError("x range must satisfy x_max > x_min.")

    return x_min, x_max

def emittance_from_covariance_2x2(covariance: np.ndarray) -> float:
    """
    Compute geometric emittance from a 2x2 covariance matrix.

    Parameters
    ----------
    covariance : np.ndarray
        2x2 covariance matrix in (u, u') space.

    Returns
    -------
    float
        Geometric emittance.
    """
    cov = np.asarray(covariance, dtype=float)
    if cov.shape != (2, 2):
        raise ValueError(f"covariance must have shape (2, 2), got {cov.shape}.")

    det = float(np.linalg.det(cov))
    if det <= 0.0:
        raise ValueError("Covariance determinant must be positive.")

    return float(np.sqrt(det))

def twiss_from_covariance_2x2(covariance: np.ndarray) -> tuple[float, float, float, float]:
    """
    Compute (alpha, beta, gamma_twiss, geometric_emittance) from a 2x2 covariance matrix.

    Parameters
    ----------
    covariance : np.ndarray
        2x2 covariance matrix in (u, u') space.

    Returns
    -------
    alpha, beta, gamma_twiss, geometric_emittance : tuple[float, float, float, float]
    """
    cov = np.asarray(covariance, dtype=float)
    if cov.shape != (2, 2):
        raise ValueError(f"covariance must have shape (2, 2), got {cov.shape}.")

    eps = emittance_from_covariance_2x2(cov)
    beta = float(cov[0, 0] / eps)
    alpha = float(-cov[0, 1] / eps)
    gamma_twiss = float(cov[1, 1] / eps)

    return alpha, beta, gamma_twiss, eps


def compute_phase_space_covariance_plane(
    dist,
    *,
    plane: Literal["x", "y"],
    weight: Union[None, str, ArrayLike] = "Q_abs",
    mask: Optional[Union[np.ndarray, Sequence[bool]]] = None,
) -> np.ndarray:
    """
    Compute the weighted 2x2 covariance matrix in one transverse plane.

    The covariance is defined in (u, u') space:
        - x plane: u = x,  u' = px / pz
        - y plane: u = y,  u' = py / pz
    """
    if plane not in {"x", "y"}:
        raise ValueError("plane must be 'x' or 'y'.")

    n = len(dist)
    w = _get_weight_array(dist, weight, absolute=True)
    m = _normalize_mask(mask, n)

    if plane == "x":
        u = _extract_data(dist, "x", n_expected=n, dtype=float, name="x")
        pu = _extract_data(dist, "px", n_expected=n, dtype=float, name="px")
    else:
        u = _extract_data(dist, "y", n_expected=n, dtype=float, name="y")
        pu = _extract_data(dist, "py", n_expected=n, dtype=float, name="py")

    pz = _extract_data(dist, "pz", n_expected=n, dtype=float, name="pz")

    valid = (
        m
        & np.isfinite(u)
        & np.isfinite(pu)
        & np.isfinite(pz)
        & np.isfinite(w)
        & (np.abs(pz) > 0.0)
    )

    u_sel = u[valid]
    pu_sel = pu[valid]
    pz_sel = pz[valid]
    w_sel = w[valid]

    if len(u_sel) < 2:
        raise ValueError("At least two valid selected particles are required.")
    if float(np.sum(w_sel)) <= 0.0:
        raise ValueError("Selected particles must have strictly positive total weight.")

    up_sel = pu_sel / pz_sel
    return _weighted_cov_2x2(u_sel, up_sel, w_sel)


def compute_phase_space_plane(
    dist,
    *,
    plane: Literal["x", "y"],
    weight: Union[None, str, ArrayLike] = "Q_abs",
    mask: Optional[Union[np.ndarray, Sequence[bool]]] = None,
    m0: float = g_m0,
) -> PhaseSpacePlaneResult:
    """
    Compute weighted transverse phase-space diagnostics for one plane.

    Parameters
    ----------
    dist : ParticleDistribution
        Input distribution.
    plane : {"x", "y"}
        Transverse plane.
    weight : None, str, or array-like
        Optional particle weights.
    mask : array-like of bool, optional
        Particle mask.
    m0 : float
        Rest mass [kg], used for normalized emittance via beta0*gamma0 = p_ref/(m0 c).

    Returns
    -------
    PhaseSpacePlaneResult
        Full transverse second-moment analysis result.
    """
    if plane not in {"x", "y"}:
        raise ValueError("plane must be 'x' or 'y'.")

    n = len(dist)
    w = _get_weight_array(dist, weight, absolute=True)
    m = _normalize_mask(mask, n)

    if plane == "x":
        u = _extract_data(dist, "x", n_expected=n, dtype=float, name="x")
        pu = _extract_data(dist, "px", n_expected=n, dtype=float, name="px")
    else:
        u = _extract_data(dist, "y", n_expected=n, dtype=float, name="y")
        pu = _extract_data(dist, "py", n_expected=n, dtype=float, name="py")

    pz = _extract_data(dist, "pz", n_expected=n, dtype=float, name="pz")
    p_abs = _extract_data(dist, "p_abs_si", n_expected=n, dtype=float, name="p_abs_si")

    valid = (
        m
        & np.isfinite(u)
        & np.isfinite(pu)
        & np.isfinite(pz)
        & np.isfinite(p_abs)
        & np.isfinite(w)
        & (np.abs(pz) > 0.0)
    )

    u_sel = u[valid]
    pu_sel = pu[valid]
    pz_sel = pz[valid]
    p_abs_sel = p_abs[valid]
    w_sel = w[valid]

    if len(u_sel) < 2:
        raise ValueError("At least two valid selected particles are required.")
    weight_sum = float(np.sum(w_sel))
    if weight_sum <= 0.0:
        raise ValueError("Selected particles must have strictly positive total weight.")

    up_sel = pu_sel / pz_sel

    mean_u = _weighted_mean(u_sel, w_sel)
    mean_up = _weighted_mean(up_sel, w_sel)

    covariance = _weighted_cov_2x2(u_sel, up_sel, w_sel)
    alpha, beta, gamma_twiss, eps_geom = twiss_from_covariance_2x2(covariance)

    # beta0 * gamma0 from the charge-weighted reference momentum (textbook
    # convention; matches MAD-X, ASTRA, Ocelot). Equivalent to the weighted
    # mean of per-particle beta*gamma by linearity, but expressed explicitly
    # as p_ref / (m0 c) to make the reference-particle convention clear.
    p_ref = _weighted_mean(p_abs_sel, w_sel)
    beta_gamma_ref = float(p_ref / (m0 * g_c))
    eps_norm = float(beta_gamma_ref * eps_geom)

    return PhaseSpacePlaneResult(
        plane=plane,
        mean_u=mean_u,
        mean_up=mean_up,
        sigma_u=float(np.sqrt(covariance[0, 0])),
        sigma_up=float(np.sqrt(covariance[1, 1])),
        cov_u_up=float(covariance[0, 1]),
        covariance=covariance,
        geometric_emittance=eps_geom,
        normalized_emittance=eps_norm,
        mean_beta_gamma=beta_gamma_ref,
        alpha=alpha,
        beta=beta,
        gamma_twiss=gamma_twiss,
        n_selected=int(len(u_sel)),
        weight_sum=weight_sum,
    )


def compute_binned_profile(
    dist: "ParticleDistribution",
    x: Union[str, ArrayLike],
    y: Union[str, ArrayLike],
    *,
    bins: int = 100,
    x_range: Optional[tuple[float, float]] = None,
    weight: Union[None, str, ArrayLike] = "Q_abs",
    stat: StatMethod = "weighted_mean",
    min_count: int = 1,
    min_weight_sum: float = 0.0,
    mask: Optional[Union[np.ndarray, Sequence[bool]]] = None,
) -> BinnedProfileResult:
    """
    Compute a 1D binned profile of y versus x.

    Parameters
    ----------
    dist : ParticleDistribution
        Source particle distribution.
    x, y : str or array-like
        Quantity keys or explicit arrays.
    bins : int
        Number of uniform bins.
    x_range : tuple[float, float], optional
        Explicit x range. If None, inferred from data.
    weight : None, str, or array-like
        Particle weights. If None, uniform weighting is used.
    stat : {"mean", "weighted_mean", "median"}
        Statistic used for y in each bin.
    min_count : int
        Minimum particle count required for a bin to be marked valid.
    min_weight_sum : float
        Minimum total weight required for a bin to be marked valid.
    mask : array-like of bool, optional
        Particle mask.

    Returns
    -------
    BinnedProfileResult
        Binned profile result.
    """
    n = len(dist)

    x_arr = _extract_data(dist, x, n_expected=n, dtype=float, name="x")
    y_arr = _extract_data(dist, y, n_expected=n, dtype=float, name="y")
    w_arr = _get_weight_array(dist, weight, absolute=True)
    m_arr = _normalize_mask(mask, n)

    valid_particle_mask = np.isfinite(x_arr) & np.isfinite(y_arr) & np.isfinite(w_arr) & m_arr
    x_use = x_arr[valid_particle_mask]
    y_use = y_arr[valid_particle_mask]
    w_use = w_arr[valid_particle_mask]

    if len(x_use) == 0:
        raise ValueError("No valid particles remain after applying finite-value filtering and mask.")

    if bins < 1:
        raise ValueError("bins must be >= 1.")

    x_min, x_max = _infer_range(x_use, x_range)
    bin_edges = np.linspace(x_min, x_max, bins + 1)
    x_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    y_stat = np.full(bins, np.nan, dtype=float)
    y_std = np.full(bins, np.nan, dtype=float)
    y_rms = np.full(bins, np.nan, dtype=float)
    counts = np.zeros(bins, dtype=int)
    weight_sum = np.zeros(bins, dtype=float)

    bin_index = _digitize_to_bins(x_use, bin_edges)

    for i in range(bins):
        sel = bin_index == i
        if not np.any(sel):
            continue

        y_bin = y_use[sel]
        w_bin = w_use[sel]

        counts[i] = len(y_bin)
        weight_sum[i] = float(np.sum(w_bin))

        if counts[i] < min_count or weight_sum[i] < min_weight_sum:
            continue

        if stat == "mean":
            y_stat[i] = float(np.mean(y_bin))
        elif stat == "weighted_mean":
            if np.sum(w_bin) <= 0.0:
                continue
            y_stat[i] = _weighted_mean(y_bin, w_bin)
        elif stat == "median":
            y_stat[i] = float(np.median(y_bin))
        else:
            raise ValueError(f"Unknown stat={stat!r}.")

        if np.sum(w_bin) > 0.0:
            y_std[i] = _weighted_std(y_bin, w_bin)
            y_rms[i] = _weighted_rms(y_bin, w_bin)

    valid_mask = np.isfinite(y_stat)

    return BinnedProfileResult(
        x_centers=x_centers,
        y_stat=y_stat,
        y_std=y_std,
        y_rms=y_rms,
        counts=counts,
        weight_sum=weight_sum,
        valid_mask=valid_mask,
        bin_edges=bin_edges,
        x_key=x if isinstance(x, str) else None,
        y_key=y if isinstance(y, str) else None,
    )


def fit_trend_from_profile(
    profile: BinnedProfileResult,
    *,
    method: FitMethod = "spline",
    poly_order: int = 1,
    poly_weight: Literal["count", "sqrt_count", "weight_sum", "sqrt_weight_sum", "none"] = "weight_sum",
    spline_smoothing: float = 0.0,
    spline_weight: Literal["count", "sqrt_count", "weight_sum", "sqrt_weight_sum", "none"] = "sqrt_count",
) -> TrendFitResult:
    """
    Fit a smooth trend using a binned profile.

    Parameters
    ----------
    profile : BinnedProfileResult
        Input binned profile.
    method : {"linear", "poly", "spline"}
        Fit method.
    poly_order : int
        Polynomial order for "poly". Ignored for "linear" and "spline".
    poly_weight : {"count", "sqrt_count", "weight_sum", "sqrt_weight_sum", "none"}
        Weighting rule for "linear" and "poly" fitting. Passed as the ``w``
        argument to ``numpy.polyfit``. Defaults to ``"weight_sum"`` so that
        bins with more charge have greater influence on the fit.
    spline_smoothing : float
        Spline smoothing strength. Passed as s = len(x) * spline_smoothing.
    spline_weight : {"count", "sqrt_count", "weight_sum", "sqrt_weight_sum", "none"}
        Weighting rule for spline fitting.

    Returns
    -------
    TrendFitResult
        Trend fit result.
    """
    x_fit = profile.x_valid
    y_fit = profile.y_valid

    if len(x_fit) < 2:
        raise ValueError("At least two valid points are required for trend fitting.")

    def _resolve_poly_weight(rule: str) -> Optional[np.ndarray]:
        if rule == "count":
            return profile.counts_valid.astype(float)
        elif rule == "sqrt_count":
            return np.sqrt(profile.counts_valid.astype(float))
        elif rule == "weight_sum":
            return profile.weight_sum_valid.astype(float)
        elif rule == "sqrt_weight_sum":
            return np.sqrt(profile.weight_sum_valid.astype(float))
        elif rule == "none":
            return None
        else:
            raise ValueError(f"Unknown poly_weight={rule!r}.")

    if method == "linear":
        w_fit = _resolve_poly_weight(poly_weight)
        coeff = np.polyfit(x_fit, y_fit, deg=1, w=w_fit)
        poly = np.poly1d(coeff)

        return TrendFitResult(
            method="linear",
            evaluator=lambda x: poly(np.asarray(x, dtype=float)),
            x_sample=x_fit,
            y_sample=y_fit,
            fit_params={"coeff": coeff},
            profile=profile,
        )

    if method == "poly":
        if poly_order < 0:
            raise ValueError("poly_order must be >= 0.")
        if len(x_fit) < poly_order + 1:
            raise ValueError(
                f"Need at least {poly_order + 1} valid points for poly_order={poly_order}."
            )

        w_fit = _resolve_poly_weight(poly_weight)
        coeff = np.polyfit(x_fit, y_fit, deg=poly_order, w=w_fit)
        poly = np.poly1d(coeff)

        return TrendFitResult(
            method="poly",
            evaluator=lambda x: poly(np.asarray(x, dtype=float)),
            x_sample=x_fit,
            y_sample=y_fit,
            fit_params={"coeff": coeff, "order": poly_order},
            profile=profile,
        )

    if method == "spline":
        if len(x_fit) < 3:
            raise ValueError("At least three valid points are required for spline fitting.")

        if spline_weight == "count":
            w_fit = profile.counts_valid.astype(float)
        elif spline_weight == "sqrt_count":
            w_fit = np.sqrt(profile.counts_valid.astype(float))
        elif spline_weight == "weight_sum":
            w_fit = profile.weight_sum_valid.astype(float)
        elif spline_weight == "sqrt_weight_sum":
            w_fit = np.sqrt(profile.weight_sum_valid.astype(float))
        elif spline_weight == "none":
            w_fit = None
        else:
            raise ValueError(f"Unknown spline_weight={spline_weight!r}.")

        spline = UnivariateSpline(
            x_fit,
            y_fit,
            w=w_fit,
            s=len(x_fit) * float(spline_smoothing),
        )

        return TrendFitResult(
            method="spline",
            evaluator=lambda x: spline(np.asarray(x, dtype=float)),
            x_sample=x_fit,
            y_sample=y_fit,
            fit_params={
                "spline": spline,
                "smoothing": spline_smoothing,
                "weight_mode": spline_weight,
            },
            profile=profile,
        )

    raise ValueError(f"Unknown method={method!r}.")


def fit_trend(
    dist: "ParticleDistribution",
    x: Union[str, ArrayLike],
    y: Union[str, ArrayLike],
    *,
    bins: int = 100,
    x_range: Optional[tuple[float, float]] = None,
    weight: Union[None, str, ArrayLike] = "Q_abs",
    stat: StatMethod = "weighted_mean",
    min_count: int = 1,
    min_weight_sum: float = 0.0,
    mask: Optional[Union[np.ndarray, Sequence[bool]]] = None,
    method: FitMethod = "spline",
    poly_order: int = 1,
    spline_smoothing: float = 0.0,
    spline_weight: Literal["count", "sqrt_count", "weight_sum", "sqrt_weight_sum", "none"] = "sqrt_count",
) -> TrendFitResult:
    """
    Compute a binned profile and fit a trend y(x) from it.
    """
    profile = compute_binned_profile(
        dist,
        x,
        y,
        bins=bins,
        x_range=x_range,
        weight=weight,
        stat=stat,
        min_count=min_count,
        min_weight_sum=min_weight_sum,
        mask=mask,
    )

    return fit_trend_from_profile(
        profile,
        method=method,
        poly_order=poly_order,
        spline_smoothing=spline_smoothing,
        spline_weight=spline_weight,
    )


def evaluate_residuals(
    dist: "ParticleDistribution",
    x: Union[str, ArrayLike],
    y: Union[str, ArrayLike],
    trend: Union[TrendFitResult, Callable[[ArrayLike], ArrayLike]],
    *,
    weight: Union[None, str, ArrayLike] = "Q_abs",
    mask: Optional[Union[np.ndarray, Sequence[bool]]] = None,
) -> ResidualAnalysisResult:
    """
    Evaluate residuals y - f(x) relative to a supplied trend.

    Parameters
    ----------
    dist : ParticleDistribution
        Source particle distribution.
    x, y : str or array-like
        Quantity keys or explicit arrays.
    trend : TrendFitResult or callable
        Fitted trend object or arbitrary callable.
    weight : None, str, or array-like
        Particle weights.
    mask : array-like of bool, optional
        Particle mask.

    Returns
    -------
    ResidualAnalysisResult
        Residual-analysis result.
    """
    n = len(dist)

    x_arr = _extract_data(dist, x, n_expected=n, dtype=float, name="x")
    y_arr = _extract_data(dist, y, n_expected=n, dtype=float, name="y")
    w_arr = _get_weight_array(dist, weight, absolute=True)
    m_arr = _normalize_mask(mask, n)

    valid_particle_mask = np.isfinite(x_arr) & np.isfinite(y_arr) & np.isfinite(w_arr) & m_arr
    x_use = x_arr[valid_particle_mask]
    y_use = y_arr[valid_particle_mask]
    w_use = w_arr[valid_particle_mask]

    if len(x_use) == 0:
        raise ValueError("No valid particles remain after applying finite-value filtering and mask.")

    if isinstance(trend, TrendFitResult):
        y_fit = trend(x_use)
    else:
        y_fit = np.asarray(trend(x_use), dtype=float)

    residual = y_use - y_fit

    if np.sum(w_use) <= 0.0:
        raise ValueError("Residual analysis requires strictly positive total weight.")

    return ResidualAnalysisResult(
        x=x_use,
        y=y_use,
        y_fit=y_fit,
        residual=residual,
        weighted_mean=_weighted_mean(residual, w_use),
        weighted_std=_weighted_std(residual, w_use),
        weighted_rms=_weighted_rms(residual, w_use),
    )


def fit_linear_chirp(
    dist: "ParticleDistribution",
    *,
    z_key: str = "z",
    pz_key: str = "pz",
    weight: Union[None, str, ArrayLike] = "Q_abs",
    mask: Optional[Union[np.ndarray, Sequence[bool]]] = None,
    bins: Optional[int] = None,
    x_range: Optional[tuple[float, float]] = None,
    use_binned_profile: bool = False,
    stat: StatMethod = "weighted_mean",
    min_count: int = 1,
    min_weight_sum: float = 0.0,
) -> TrendFitResult:
    """
    Fit a linear longitudinal chirp pz(z).

    Parameters
    ----------
    dist : ParticleDistribution
        Source distribution.
    z_key : str
        Quantity key for longitudinal coordinate.
    pz_key : str
        Quantity key for longitudinal momentum.
    weight : None, str, or array-like
        Particle weights.
    mask : array-like of bool, optional
        Particle mask.
    bins : int, optional
        Number of bins if use_binned_profile=True.
    x_range : tuple[float, float], optional
        Optional fitting range.
    use_binned_profile : bool
        If False, fit directly on particle data.
        If True, first compute a binned profile and then fit linearly.
    stat : {"mean", "weighted_mean", "median"}
        Per-bin statistic if use_binned_profile=True.
    min_count : int
        Minimum particle count per valid bin if use_binned_profile=True.
    min_weight_sum : float
        Minimum total weight per valid bin if use_binned_profile=True.

    Returns
    -------
    TrendFitResult
        Linear fit result.
    """
    if use_binned_profile:
        if bins is None:
            bins = 100
        return fit_trend(
            dist,
            z_key,
            pz_key,
            bins=bins,
            x_range=x_range,
            weight=weight,
            stat=stat,
            min_count=min_count,
            min_weight_sum=min_weight_sum,
            mask=mask,
            method="linear",
        )

    n = len(dist)
    z_arr = _extract_data(dist, z_key, n_expected=n, dtype=float, name=z_key)
    pz_arr = _extract_data(dist, pz_key, n_expected=n, dtype=float, name=pz_key)
    w_arr = _get_weight_array(dist, weight, absolute=True)
    m_arr = _normalize_mask(mask, n)

    valid = np.isfinite(z_arr) & np.isfinite(pz_arr) & np.isfinite(w_arr) & m_arr
    z_use = z_arr[valid]
    pz_use = pz_arr[valid]
    w_use = w_arr[valid]

    if x_range is not None:
        z_min, z_max = x_range
        range_mask = (z_use >= z_min) & (z_use <= z_max)
        z_use = z_use[range_mask]
        pz_use = pz_use[range_mask]
        w_use = w_use[range_mask]

    if len(z_use) < 2:
        raise ValueError("At least two valid particles are required for linear chirp fitting.")

    coeff = np.polyfit(z_use, pz_use, deg=1, w=w_use)
    poly = np.poly1d(coeff)

    return TrendFitResult(
        method="linear",
        evaluator=lambda x: poly(np.asarray(x, dtype=float)),
        x_sample=z_use,
        y_sample=pz_use,
        fit_params={
            "coeff": coeff,
            "slope": float(coeff[0]),
            "intercept": float(coeff[1]),
        },
        profile=None,
    )


def analyze_longitudinal_trend(
    dist: "ParticleDistribution",
    *,
    z_key: str = "z",
    pz_key: str = "pz",
    bins: int = 100,
    weight: Union[None, str, ArrayLike] = "Q_abs",
    stat: StatMethod = "weighted_mean",
    min_count: int = 1,
    min_weight_sum: float = 0.0,
    mask: Optional[Union[np.ndarray, Sequence[bool]]] = None,
    method: FitMethod = "spline",
    poly_order: int = 1,
    spline_smoothing: float = 0.0,
    spline_weight: Literal["count", "sqrt_count", "weight_sum", "sqrt_weight_sum", "none"] = "sqrt_count",
) -> AnalyzeLongitudinalTrendResult:
    """
    Convenience wrapper for longitudinal pz(z) analysis.

    Returns
    -------
    AnalyzeLongitudinalTrendResult
        Frozen dataclass with fields:

        - ``profile``: :class:`BinnedProfileResult`
        - ``trend``: :class:`TrendFitResult`
        - ``residuals``: :class:`ResidualAnalysisResult`
    """
    profile = compute_binned_profile(
        dist,
        z_key,
        pz_key,
        bins=bins,
        weight=weight,
        stat=stat,
        min_count=min_count,
        min_weight_sum=min_weight_sum,
        mask=mask,
    )

    trend = fit_trend_from_profile(
        profile,
        method=method,
        poly_order=poly_order,
        spline_smoothing=spline_smoothing,
        spline_weight=spline_weight,
    )

    residuals = evaluate_residuals(
        dist,
        z_key,
        pz_key,
        trend,
        weight=weight,
        mask=mask,
    )

    return AnalyzeLongitudinalTrendResult(
        profile=profile,
        trend=trend,
        residuals=residuals,
    )

def compute_longitudinal_linearity(
    dist: "ParticleDistribution",
    *,
    z_key: str = "z",
    pz_key: str = "pz",
    bins: int = 100,
    weight: Union[None, str, ArrayLike] = "Q_abs",
    stat: StatMethod = "weighted_mean",
    min_count: int = 1,
    min_weight_sum: float = 0.0,
    mask: Optional[Union[np.ndarray, Sequence[bool]]] = None,
) -> LongitudinalLinearityResult:
    """
    Quantify the linearity of the longitudinal phase space pz(z).

    The metric is based on the binned chirp profile, so it reflects the
    nonlinearity of the chirp itself and is independent of slice energy spread.

    The nonlinear fraction is defined as::

        f_nonlinear = RMS_w(pz_bin - linear_fit(z_bin)) / std_w_particles(pz)

    where the numerator is the charge-weighted RMS of bin-level residuals from
    a linear fit, and the denominator is the charge-weighted standard deviation
    of the per-particle pz (total energy spread). This normalization remains
    well-defined even when the chirp slope is zero (flat phase space), in which
    case the residuals are near zero and f_nonlinear ≈ 0.

    Parameters
    ----------
    dist : ParticleDistribution
        Input distribution.
    z_key : str
        Quantity key for longitudinal position.
    pz_key : str
        Quantity key for longitudinal momentum.
    bins : int
        Number of bins for the pz(z) profile.
    weight : None, str, or array-like
        Particle weights for the binned profile.
    stat : {"mean", "weighted_mean", "median"}
        Per-bin statistic.
    min_count : int
        Minimum particle count for a bin to be considered valid.
    min_weight_sum : float
        Minimum total weight for a bin to be considered valid.
    mask : array-like of bool, optional
        Particle mask.

    Returns
    -------
    LongitudinalLinearityResult
        Contains f_nonlinear, r_squared, the binned profile, and the fitted
        linear trend.
    """
    profile = compute_binned_profile(
        dist,
        z_key,
        pz_key,
        bins=bins,
        weight=weight,
        stat=stat,
        min_count=min_count,
        min_weight_sum=min_weight_sum,
        mask=mask,
    )

    linear_trend = fit_trend_from_profile(profile, method="linear")

    x_valid = profile.x_valid
    y_valid = profile.y_valid
    w_valid = profile.weight_sum[profile.valid_mask]

    residuals = y_valid - linear_trend(x_valid)

    sigma_nonlinear = _weighted_rms(residuals, w_valid)

    pz_arr = _extract_data(dist, pz_key, n_expected=len(dist), dtype=float, name=pz_key)
    w_arr = _get_weight_array(dist, weight, absolute=True)
    m_arr = _normalize_mask(mask, len(dist))
    valid_particles = np.isfinite(pz_arr) & np.isfinite(w_arr) & m_arr
    sigma_total = _weighted_std(pz_arr[valid_particles], w_arr[valid_particles])

    if sigma_total == 0.0:
        raise ValueError("pz has zero spread; cannot compute linearity metric.")

    f_nonlinear = float(sigma_nonlinear / sigma_total)

    return LongitudinalLinearityResult(
        f_nonlinear=f_nonlinear,
        r_squared=float(1.0 - f_nonlinear**2),
        profile=profile,
        linear_trend=linear_trend,
    )


def _compute_charge_histogram(
    z: np.ndarray,
    q: np.ndarray,
    bins: int,
    z_range: Optional[tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (edges, charge_hist, lambda_z) for a z-weighted charge histogram."""
    charge_hist, edges = np.histogram(z, bins=bins, range=z_range, weights=q)
    dz = np.diff(edges)
    if np.any(dz <= 0.0):
        raise ValueError("Bin edges must be strictly increasing.")
    lambda_z = charge_hist / dz  # [C/m]
    return edges, charge_hist, lambda_z


def line_charge_profile_z(
    dist: ParticleDistribution,
    *,
    bins: int = 100,
    range: Optional[tuple[float, float]] = None,
    z_key: str = "z",
    q_key: str = "Q",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the longitudinal line-charge-density profile lambda(z).

    Parameters
    ----------
    dist
        ParticleDistribution instance.
    bins
        Number of bins.
    range
        Optional z range passed to numpy.histogram.
    z_key
        Quantity key for longitudinal position, usually 'z'.
    q_key
        Quantity key for particle charge, usually 'Q'.

    Returns
    -------
    z_centers, lambda_z
        z_centers:
            Bin centers.
        lambda_z:
            Line charge density in [C/m], one value per bin.

    Notes
    -----
    The line charge density is computed as

        lambda_k = Q_k / dz_k

    where Q_k is the total absolute charge in bin k and dz_k is the bin width.
    """
    z = np.asarray(dist.get_data(z_key), dtype=float)
    q = np.abs(np.asarray(dist.get_data(q_key), dtype=float))
    edges, _, lambda_z = _compute_charge_histogram(z, q, bins, range)
    z_centers = 0.5 * (edges[:-1] + edges[1:])
    return z_centers, lambda_z


def current_profile_z(
    dist: ParticleDistribution,
    *,
    bins: Optional[int] = None,
    range: Optional[tuple[float, float]] = None,
    z_key: str = "z",
    q_key: str = "Q",
    vz_key: str = "vz",
    use_c_approx: bool = False,
    smooth: bool = False,
    smooth_window: int = 17,
    smooth_poly: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the longitudinal current profile I(z).

    Parameters
    ----------
    dist
        ParticleDistribution instance.
    bins
        Number of bins.  If ``None`` (default), the bin count is chosen
        automatically as ``clip(N // 500, 10, 500)`` where N is the number of
        macroparticles, ensuring roughly 500 particles per bin on average.
    range
        Optional z range passed to numpy.histogram.
    z_key
        Quantity key for longitudinal position, usually 'z'.
    q_key
        Quantity key for particle charge, usually 'Q'.
    vz_key
        Quantity key for longitudinal velocity, usually 'vz'.
    use_c_approx
        If True, use I = lambda * c.
        If False, use the bin-wise charge-weighted mean vz.
    smooth
        If True, apply Savitzky-Golay smoothing to the line charge density
        before computing the current.  Total charge is preserved by rescaling.
    smooth_window
        Window length for Savitzky-Golay filter (must be odd and >= 3).
        Automatically clamped to the number of bins if necessary.  Default 17.
    smooth_poly
        Polynomial order for Savitzky-Golay filter.  Default 4.

    Returns
    -------
    z_centers, current
        z_centers:
            Bin centers.
        current:
            Current profile in [A], one value per bin.

    Notes
    -----
    The current is computed as

        I(z) = lambda(z) * v_z

    where lambda(z) is the line charge density [C/m].

    If `use_c_approx=True`, then

        I(z) = lambda(z) * c

    which is often a good approximation for relativistic beams.
    """
    z = np.asarray(dist.get_data(z_key), dtype=float)
    q = np.abs(np.asarray(dist.get_data(q_key), dtype=float))
    if bins is None:
        bins = int(np.clip(len(z) // 500, 10, 500))
    edges, charge_hist, lambda_z = _compute_charge_histogram(z, q, bins, range)

    if smooth:
        from scipy.signal import savgol_filter
        n = len(lambda_z)
        win = min(smooth_window, n)
        win = win if win % 2 == 1 else win - 1
        win = max(win, 3)
        poly = min(smooth_poly, win - 1)
        smoothed = savgol_filter(lambda_z, win, poly)
        smoothed = np.maximum(smoothed, 0.0)
        total_orig = float(np.sum(charge_hist))
        total_smooth = float(np.sum(smoothed * np.diff(edges)))
        if total_smooth > 0:
            smoothed *= total_orig / total_smooth
        lambda_z = smoothed

    if use_c_approx:
        v_rep = np.full_like(lambda_z, g_c, dtype=float)
    else:
        vz = np.asarray(dist.get_data(vz_key), dtype=float)
        qvz_hist, _ = np.histogram(z, bins=edges, weights=q * vz)
        with np.errstate(divide="ignore", invalid="ignore"):
            v_rep = np.divide(
                qvz_hist,
                charge_hist,
                out=np.zeros_like(qvz_hist, dtype=float),
                where=charge_hist != 0.0,
            )

    z_centers = 0.5 * (edges[:-1] + edges[1:])
    return z_centers, lambda_z * v_rep  # [A]


# ---------------------------------------------------------------------------
# Current-profile fitting
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CurrentProfileFitResult:
    """Result of a 1-D current-profile fit."""
    profile:   str           # "gaussian" or "parabola"
    z0:        float         # fitted centre [m]
    sigma:     float         # RMS-equivalent width [m]
    amplitude: float         # fitted peak current [A]
    z_curve:   np.ndarray    # dense z array for plotting [m]
    I_curve:   np.ndarray    # fitted I(z) values [A]
    success:   bool          # True if scipy.optimize converged


def fit_current_profile(
    z_centers: np.ndarray,
    I_z: np.ndarray,
    profile: str = "gaussian",
    *,
    fix_peak: bool = True,
    fit_threshold: float = 0.05,
    fit_weights: str = "uniform",
    n_curve: int = 500,
) -> CurrentProfileFitResult:
    """
    Fit a 1-D model to a current profile I(z).

    Parameters
    ----------
    z_centers
        Bin-centre positions [m].
    I_z
        Current values at each bin [A].
    profile
        ``"gaussian"``  — fits A · exp(−(z−z₀)²/(2σ²)).
        ``"parabola"``  — fits A · max(1−((z−z₀)/w)², 0); the returned
        *sigma* is the RMS-equivalent width w/√3.
    fix_peak
        If ``True`` (default), the amplitude A and centre z₀ are fixed to
        the observed peak of *I_z* before fitting; only the width σ (or w)
        is a free parameter.  This anchors the fitted curve to the actual
        peak and avoids tail-driven shifts of A and z₀.
        If ``False``, all three parameters are fitted freely.
    fit_threshold
        Bins below ``fit_threshold · max(I_z)`` are excluded from the
        fit to suppress tail noise.  Default 0.05.
    fit_weights
        Weighting scheme for the least-squares fit (ignored when
        *fix_peak* is ``True`` with only one free parameter).

        ``"uniform"``  — equal weights (standard OLS).
        ``"current"``  — weight ∝ I(z), i.e. ``sigma = 1/sqrt(I+ε)``;
                         residuals near the peak dominate.  Default.
        ``"current_sq"`` — weight ∝ I(z)², i.e. ``sigma = 1/(I+ε)``;
                           even stronger peak sensitivity.
    n_curve
        Number of points in the returned smooth curve.  Default 500.

    Returns
    -------
    CurrentProfileFitResult
    """
    from scipy.optimize import curve_fit

    if profile not in ("gaussian", "parabola"):
        raise ValueError(f"profile must be 'gaussian' or 'parabola', got {profile!r}.")
    if fit_weights not in ("uniform", "current", "current_sq"):
        raise ValueError(
            f"fit_weights must be 'uniform', 'current', or 'current_sq', "
            f"got {fit_weights!r}."
        )

    z_centers = np.asarray(z_centers, dtype=float)
    I_z       = np.asarray(I_z,       dtype=float)

    if z_centers.shape != I_z.shape:
        raise ValueError(
            f"z_centers and I_z must have the same shape; got {z_centers.shape} vs {I_z.shape}."
        )
    if z_centers.size == 0:
        raise ValueError("fit_current_profile received empty z_centers / I_z arrays.")
    if z_centers.size < 3:
        raise ValueError(
            f"fit_current_profile needs at least 3 bins to fit a 3-parameter profile; got {z_centers.size}."
        )

    I_peak    = float(I_z.max())

    # Initial guesses from weighted moments.
    safe_I = np.maximum(I_z, 0.0)
    W = safe_I.sum()
    if W > 0:
        z0_guess  = float(np.dot(safe_I, z_centers) / W)
        sig_guess = float(np.sqrt(np.dot(safe_I, (z_centers - z0_guess) ** 2) / W))
    else:
        z0_guess  = float(z_centers.mean())
        sig_guess = float((z_centers[-1] - z_centers[0]) / 4)
    if sig_guess == 0:
        sig_guess = float((z_centers[-1] - z_centers[0]) / 4) or 1e-6

    # When fix_peak=True, anchor A and z0 to the observed peak.
    peak_idx = int(np.argmax(I_z))
    amp_fixed = I_peak
    z0_fixed  = float(z_centers[peak_idx])

    fit_mask = I_z >= fit_threshold * I_peak
    zf, If = z_centers[fit_mask], I_z[fit_mask]

    # Build sigma array for weighted least squares.
    eps = 1e-6 * max(I_peak, 1.0)
    if fit_weights == "uniform":
        sigma_w = None
    elif fit_weights == "current":
        sigma_w = 1.0 / np.sqrt(np.maximum(If, 0.0) + eps)
    else:  # current_sq
        sigma_w = 1.0 / (np.maximum(If, 0.0) + eps)

    success = True
    try:
        if profile == "gaussian":
            if fix_peak:
                def _gauss_w(z_, sig):
                    return amp_fixed * np.exp(-0.5 * ((z_ - z0_fixed) / sig) ** 2)
                popt, _ = curve_fit(
                    _gauss_w, zf, If,
                    p0=[sig_guess],
                    sigma=sigma_w, absolute_sigma=False,
                    bounds=([1e-12], [np.inf]),
                    maxfev=5000,
                )
                z0_fit, sig_fit, amp_fit = z0_fixed, abs(float(popt[0])), amp_fixed
            else:
                def _gauss(z_, A, z0, sig):
                    return A * np.exp(-0.5 * ((z_ - z0) / sig) ** 2)
                popt, _ = curve_fit(
                    _gauss, zf, If,
                    p0=[I_peak, z0_guess, sig_guess],
                    sigma=sigma_w, absolute_sigma=False,
                    bounds=([0, z_centers.min(), 1e-12],
                            [np.inf, z_centers.max(), np.inf]),
                    maxfev=5000,
                )
                z0_fit, sig_fit, amp_fit = float(popt[1]), abs(float(popt[2])), float(popt[0])

        else:  # parabola
            if fix_peak:
                def _parabola_w(z_, w_):
                    return amp_fixed * np.maximum(
                        1.0 - ((z_ - z0_fixed) / w_) ** 2, 0.0)
                popt, _ = curve_fit(
                    _parabola_w, zf, If,
                    p0=[sig_guess * np.sqrt(3)],
                    sigma=sigma_w, absolute_sigma=False,
                    bounds=([1e-12], [np.inf]),
                    maxfev=5000,
                )
                z0_fit  = z0_fixed
                sig_fit = abs(float(popt[0])) / np.sqrt(3)
                amp_fit = amp_fixed
            else:
                def _parabola(z_, A, z0, w_):
                    return A * np.maximum(1.0 - ((z_ - z0) / w_) ** 2, 0.0)
                popt, _ = curve_fit(
                    _parabola, zf, If,
                    p0=[I_peak, z0_guess, sig_guess * np.sqrt(3)],
                    sigma=sigma_w, absolute_sigma=False,
                    bounds=([0, z_centers.min(), 1e-12],
                            [np.inf, z_centers.max(), np.inf]),
                    maxfev=5000,
                )
                z0_fit  = float(popt[1])
                sig_fit = abs(float(popt[2])) / np.sqrt(3)
                amp_fit = float(popt[0])

    except RuntimeError:
        success  = False
        z0_fit   = z0_fixed if fix_peak else z0_guess
        sig_fit  = sig_guess
        amp_fit  = amp_fixed if fix_peak else I_peak

    z_dense = np.linspace(z_centers[0], z_centers[-1], n_curve)
    if profile == "gaussian":
        I_dense = amp_fit * np.exp(-0.5 * ((z_dense - z0_fit) / sig_fit) ** 2)
    else:
        I_dense = amp_fit * np.maximum(1.0 - ((z_dense - z0_fit) / (sig_fit * np.sqrt(3))) ** 2, 0.0)

    return CurrentProfileFitResult(
        profile=profile,
        z0=z0_fit,
        sigma=sig_fit,
        amplitude=amp_fit,
        z_curve=z_dense,
        I_curve=I_dense,
        success=success,
    )


# ---------------------------------------------------------------------------
# Unified beam diagnostics
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BeamDiagnosticsResult:
    """
    Comprehensive beam statistics returned by :func:`compute_beam_diagnostics`.

    All quantities are in SI base units (m, rad, eV, A, C) unless noted.
    Fields related to particle survival are ``None`` when the input
    distribution carries no ``status`` quantity.
    """

    # ── particle count & charge ───────────────────────────────────────
    n_total:             int     # total macroparticles (incl. lost)
    n_alive:             int     # alive macroparticles used for physics
    n_lost:              Optional[int]    # None → no status available
    Q_total_C:           float            # total |charge| [C]
    Q_alive_C:           float            # alive |charge| [C]
    Q_lost_C:            Optional[float]  # None → no status available

    # ── relativistic ─────────────────────────────────────────────────
    gamma0:              float   # mean Lorentz factor
    beta0:               float   # mean relativistic beta
    mean_E_kin_eV:       float   # mean kinetic energy [eV]
    sig_E_kin_eV:        float   # RMS energy spread [eV]
    sig_E_rel:           float   # sig_E / E_mean (dimensionless)

    # ── energy spread decomposition ───────────────────────────────────
    sig_E_corr_eV:       float   # linear correlated energy spread [eV]
    sig_E_uncorr_eV:     float   # slice-based uncorrelated energy spread [eV]
    sig_E_slice_max_eV:  float   # max single-slice energy spread [eV]

    # ── transverse x ─────────────────────────────────────────────────
    mean_x:              float   # [m]
    sig_x:               float   # [m]
    mean_xp:             float   # [rad]
    sig_xp:              float   # [rad]
    emit_x:              float   # geometric emittance [m·rad]
    nemit_x:             float   # normalised emittance [m·rad]
    alpha_x:             float
    beta_x:              float   # Twiss beta [m]

    # ── transverse y ─────────────────────────────────────────────────
    mean_y:              float   # [m]
    sig_y:               float   # [m]
    mean_yp:             float   # [rad]
    sig_yp:              float   # [rad]
    emit_y:              float   # geometric emittance [m·rad]
    nemit_y:             float   # normalised emittance [m·rad]
    alpha_y:             float
    beta_y:              float   # Twiss beta [m]

    # ── longitudinal ─────────────────────────────────────────────────
    mean_z:              float   # [m]
    sig_z:               float   # [m]
    chirp:               float   # dδ/dz [m⁻¹]
    quadratic_chirp:     float   # [m⁻²]
    cubic_chirp:         float   # [m⁻³]

    # ── current ───────────────────────────────────────────────────────
    I_peak_raw:          float   # [A]
    I_peak_smooth:       float   # [A]

    def to_dict(self) -> Dict[str, object]:
        """Return all fields as a plain dictionary."""
        return asdict(self)


def _compute_slice_energy_spread(
    dist: "ParticleDistribution",
    n_slices: int,
) -> tuple[float, float]:
    """Return (sig_E_uncorr_eV, sig_E_slice_max_eV) via slice decomposition."""
    z = dist.get_data("z").astype(float)
    e = dist.get_data("kinetic_energy_eV").astype(float)
    w = np.abs(dist.get_data("Q").astype(float))
    edges = np.linspace(z.min(), z.max(), n_slices + 1)
    slice_stats: list[tuple[float, float]] = []  # (variance, weight)
    max_sig = 0.0
    for i in range(n_slices):
        mask = (z >= edges[i]) & (z < edges[i + 1])
        if mask.sum() < 5:
            continue
        wi = w[mask]
        W = float(wi.sum())
        if W == 0.0:
            continue
        ei = e[mask]
        mean_e = float(np.average(ei, weights=wi))
        var = float(np.average((ei - mean_e) ** 2, weights=wi))
        slice_stats.append((var, W))
        sig = float(np.sqrt(var))
        if sig > max_sig:
            max_sig = sig
    if not slice_stats:
        return 0.0, 0.0
    total_W = sum(W for _, W in slice_stats)
    uncorr = float(np.sqrt(sum(v * W for v, W in slice_stats) / total_W))
    return uncorr, max_sig


def compute_beam_diagnostics(
    dist: "ParticleDistribution",
    *,
    n_slices_energy: int = 50,
) -> BeamDiagnosticsResult:
    """
    Compute a comprehensive set of beam statistics and return them as a
    :class:`BeamDiagnosticsResult`.

    If the distribution carries a ``status`` quantity (as written by Astra),
    particles with ``status < 0`` are treated as lost: survival statistics
    are computed from the full distribution and all physics quantities are
    evaluated on the alive subset only.  Without ``status``, all particles
    are treated as alive and the survival fields are ``None``.

    Parameters
    ----------
    dist
        Input distribution (may include lost particles with status < 0).
    n_slices_energy
        Number of z slices for slice energy-spread decomposition.
        Default 50.

    Returns
    -------
    BeamDiagnosticsResult
    """
    # ── survival split ────────────────────────────────────────────────
    if "status" in dist.extra_quantity_keys:
        mask_alive = dist.get_data("status").astype(int) >= 0
        d = dist.slice(mask_alive)
        n_lost   = int((~mask_alive).sum())
        Q_lost_C = float(np.abs(dist.get_data("Q")[~mask_alive]).sum())
        n_lost_out:  Optional[int]   = n_lost
        Q_lost_out:  Optional[float] = Q_lost_C
    else:
        d = dist
        n_lost_out  = None
        Q_lost_out  = None

    n_total  = len(dist.get_data("x"))
    n_alive  = len(d.get_data("x"))
    Q_total  = float(np.abs(dist.get_data("Q")).sum())
    Q_alive  = float(d.total_charge_abs)

    # ── relativistic ─────────────────────────────────────────────────
    mean_E   = d.mean("kinetic_energy_eV")
    sig_E    = d.std("kinetic_energy_eV")

    # ── energy spread decomposition ───────────────────────────────────
    sig_E_uncorr, sig_E_slice_max = _compute_slice_energy_spread(d, n_slices_energy)

    # ── transverse ───────────────────────────────────────────────────
    _, I_smo = d.current_profile_z_smooth

    return BeamDiagnosticsResult(
        # particle count & charge
        n_total          = n_total,
        n_alive          = n_alive,
        n_lost           = n_lost_out,
        Q_total_C        = Q_total,
        Q_alive_C        = Q_alive,
        Q_lost_C         = Q_lost_out,
        # relativistic
        gamma0           = d.gamma0,
        beta0            = d.beta0,
        mean_E_kin_eV    = mean_E,
        sig_E_kin_eV     = sig_E,
        sig_E_rel        = sig_E / mean_E if mean_E != 0.0 else 0.0,
        # energy spread decomposition
        sig_E_corr_eV    = d.cor_ekin,
        sig_E_uncorr_eV  = sig_E_uncorr,
        sig_E_slice_max_eV = sig_E_slice_max,
        # transverse x
        mean_x           = d.mean("x"),
        sig_x            = d.std("x"),
        mean_xp          = d.mean("xp"),
        sig_xp           = d.std("xp"),
        emit_x           = d.emit_x,
        nemit_x          = d.nemit_x,
        alpha_x          = d.alpha_x,
        beta_x           = d.beta_x,
        # transverse y
        mean_y           = d.mean("y"),
        sig_y            = d.std("y"),
        mean_yp          = d.mean("yp"),
        sig_yp           = d.std("yp"),
        emit_y           = d.emit_y,
        nemit_y          = d.nemit_y,
        alpha_y          = d.alpha_y,
        beta_y           = d.beta_y,
        # longitudinal
        mean_z           = d.mean("z"),
        sig_z            = d.std("z"),
        chirp            = d.chirp,
        quadratic_chirp  = d.quadratic_chirp,
        cubic_chirp      = d.cubic_chirp,
        # current
        I_peak_raw       = d.I_peak,
        I_peak_smooth    = float(np.max(I_smo)),
    )



