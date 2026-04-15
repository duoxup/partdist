from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional, Sequence, Union, TYPE_CHECKING

import numpy as np
from scipy.interpolate import UnivariateSpline

from xtils import g_m0, g_c
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
          eps_n = <beta*gamma> * eps_geom
      where <beta*gamma> is the weighted mean over the selected particles
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
    weight: Union[None, str, ArrayLike] = None,
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
    weight: Union[None, str, ArrayLike] = None,
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
        Rest mass [kg], used for normalized emittance via <p/(m0 c)>.

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

    mean_beta_gamma = _weighted_mean(p_abs_sel / (m0 * g_c), w_sel)
    eps_norm = float(mean_beta_gamma * eps_geom)

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
        mean_beta_gamma=float(mean_beta_gamma),
        alpha=alpha,
        beta=beta,
        gamma_twiss=gamma_twiss,
        n_selected=int(len(u_sel)),
        weight_sum=weight_sum,
    )


def compute_phase_space_x(
    dist,
    *,
    weight: Union[None, str, ArrayLike] = None,
    mask: Optional[Union[np.ndarray, Sequence[bool]]] = None,
    m0: float = g_m0,
) -> PhaseSpacePlaneResult:
    """
    Compute transverse phase-space diagnostics in the x plane.
    """
    return compute_phase_space_plane(dist, plane="x", weight=weight, mask=mask, m0=m0)


def compute_phase_space_y(
    dist,
    *,
    weight: Union[None, str, ArrayLike] = None,
    mask: Optional[Union[np.ndarray, Sequence[bool]]] = None,
    m0: float = g_m0,
) -> PhaseSpacePlaneResult:
    """
    Compute transverse phase-space diagnostics in the y plane.
    """
    return compute_phase_space_plane(dist, plane="y", weight=weight, mask=mask, m0=m0)


def compute_twiss_plane(
    dist,
    *,
    plane: Literal["x", "y"],
    weight: Union[None, str, ArrayLike] = None,
    mask: Optional[Union[np.ndarray, Sequence[bool]]] = None,
    m0: float = g_m0,
) -> PhaseSpacePlaneResult:
    """
    Convenience alias returning the full phase-space result for one plane.

    This keeps the old semantic entry point while using the more general result model.
    """
    return compute_phase_space_plane(dist, plane=plane, weight=weight, mask=mask, m0=m0)


def compute_twiss_x(
    dist,
    *,
    weight: Union[None, str, ArrayLike] = None,
    mask: Optional[Union[np.ndarray, Sequence[bool]]] = None,
    m0: float = g_m0,
) -> PhaseSpacePlaneResult:
    """
    Convenience alias for x-plane phase-space/Twiss analysis.
    """
    return compute_phase_space_x(dist, weight=weight, mask=mask, m0=m0)


def compute_twiss_y(
    dist,
    *,
    weight: Union[None, str, ArrayLike] = None,
    mask: Optional[Union[np.ndarray, Sequence[bool]]] = None,
    m0: float = g_m0,
) -> PhaseSpacePlaneResult:
    """
    Convenience alias for y-plane phase-space/Twiss analysis.
    """
    return compute_phase_space_y(dist, weight=weight, mask=mask, m0=m0)


def compute_geometric_emittance_plane(
    dist,
    *,
    plane: Literal["x", "y"],
    weight: Union[None, str, ArrayLike] = None,
    mask: Optional[Union[np.ndarray, Sequence[bool]]] = None,
    m0: float = g_m0,
) -> float:
    """
    Compute geometric emittance in one transverse plane.
    """
    return compute_phase_space_plane(dist, plane=plane, weight=weight, mask=mask, m0=m0).geometric_emittance


def compute_normalized_emittance_plane(
    dist,
    *,
    plane: Literal["x", "y"],
    weight: Union[None, str, ArrayLike] = None,
    mask: Optional[Union[np.ndarray, Sequence[bool]]] = None,
    m0: float = g_m0,
) -> float:
    """
    Compute normalized emittance in one transverse plane.
    """
    return compute_phase_space_plane(dist, plane=plane, weight=weight, mask=mask, m0=m0).normalized_emittance


def compute_geometric_emittance_x(
    dist,
    *,
    weight: Union[None, str, ArrayLike] = None,
    mask: Optional[Union[np.ndarray, Sequence[bool]]] = None,
    m0: float = g_m0,
) -> float:
    """
    Compute x-plane geometric emittance.
    """
    return compute_geometric_emittance_plane(dist, plane="x", weight=weight, mask=mask, m0=m0)


def compute_geometric_emittance_y(
    dist,
    *,
    weight: Union[None, str, ArrayLike] = None,
    mask: Optional[Union[np.ndarray, Sequence[bool]]] = None,
    m0: float = g_m0,
) -> float:
    """
    Compute y-plane geometric emittance.
    """
    return compute_geometric_emittance_plane(dist, plane="y", weight=weight, mask=mask, m0=m0)


def compute_normalized_emittance_x(
    dist,
    *,
    weight: Union[None, str, ArrayLike] = None,
    mask: Optional[Union[np.ndarray, Sequence[bool]]] = None,
    m0: float = g_m0,
) -> float:
    """
    Compute x-plane normalized emittance.
    """
    return compute_normalized_emittance_plane(dist, plane="x", weight=weight, mask=mask, m0=m0)


def compute_normalized_emittance_y(
    dist,
    *,
    weight: Union[None, str, ArrayLike] = None,
    mask: Optional[Union[np.ndarray, Sequence[bool]]] = None,
    m0: float = g_m0,
) -> float:
    """
    Compute y-plane normalized emittance.
    """
    return compute_normalized_emittance_plane(dist, plane="y", weight=weight, mask=mask, m0=m0)

def compute_binned_profile(
    dist: "ParticleDistribution",
    x: Union[str, ArrayLike],
    y: Union[str, ArrayLike],
    *,
    bins: int = 100,
    x_range: Optional[tuple[float, float]] = None,
    weight: Union[None, str, ArrayLike] = None,
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
    weight: Union[None, str, ArrayLike] = None,
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
    weight: Union[None, str, ArrayLike] = None,
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
    weight: Union[None, str, ArrayLike] = None,
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
    weight: Union[None, str, ArrayLike] = None,
    stat: StatMethod = "weighted_mean",
    min_count: int = 1,
    min_weight_sum: float = 0.0,
    mask: Optional[Union[np.ndarray, Sequence[bool]]] = None,
    method: FitMethod = "spline",
    poly_order: int = 1,
    spline_smoothing: float = 0.0,
    spline_weight: Literal["count", "sqrt_count", "weight_sum", "sqrt_weight_sum", "none"] = "sqrt_count",
) -> dict:
    """
    Convenience wrapper for longitudinal pz(z) analysis.

    Returns a dictionary containing:
    - "profile": BinnedProfileResult
    - "trend": TrendFitResult
    - "residuals": ResidualAnalysisResult
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

    return {
        "profile": profile,
        "trend": trend,
        "residuals": residuals,
    }

def compute_longitudinal_linearity(
    dist: "ParticleDistribution",
    *,
    z_key: str = "z",
    pz_key: str = "pz",
    bins: int = 100,
    weight: Union[None, str, ArrayLike] = None,
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
    bins: int = 100,
    range: Optional[tuple[float, float]] = None,
    z_key: str = "z",
    q_key: str = "Q",
    vz_key: str = "vz",
    use_c_approx: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the longitudinal current profile I(z).

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
    vz_key
        Quantity key for longitudinal velocity, usually 'vz'.
    use_c_approx
        If True, use I = lambda * c.
        If False, use the bin-wise charge-weighted mean vz.

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
    edges, charge_hist, lambda_z = _compute_charge_histogram(z, q, bins, range)

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







