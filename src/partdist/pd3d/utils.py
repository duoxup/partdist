from __future__ import annotations

from typing import Optional, Sequence, Union, TYPE_CHECKING

import numpy as np

from scipy.constants import c as g_c, m_e as g_m0, e as g_e0
from partdist import kinematics as relconv

if TYPE_CHECKING:
    from .core import ParticleDistribution


ArrayLike = Union[np.ndarray, Sequence[float], float]
WeightLike = Union[None, str, ArrayLike]


def _copy_or_inplace(dist: "ParticleDistribution", inplace: bool) -> "ParticleDistribution":
    """
    Return the original distribution if inplace=True, otherwise a copy.
    """
    return dist if inplace else dist.copy()


def _as_1d_array(data: ArrayLike, *, dtype=None, name: str = "array") -> np.ndarray:
    """
    Convert input to a 1D numpy array.

    Parameters
    ----------
    data : array-like
        Input data.
    dtype : optional
        Target dtype passed to np.asarray.
    name : str
        Name used in error messages.

    Returns
    -------
    np.ndarray
        Flattened 1D array.
    """
    arr = np.asarray(data, dtype=dtype)

    if arr.ndim == 0:
        arr = arr.reshape(1)
    else:
        arr = arr.reshape(-1)

    return arr


def _validate_length(arr: np.ndarray, n: int, *, name: str = "array") -> None:
    """
    Validate that a 1D array has the expected length.
    """
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array, got ndim={arr.ndim}.")
    if len(arr) != n:
        raise ValueError(f"{name} must have length {n}, got {len(arr)}.")


def _normalize_mask(
    mask: Optional[Union[np.ndarray, Sequence[bool]]],
    n: int,
) -> np.ndarray:
    """
    Normalize a mask to a boolean array of length n.

    Parameters
    ----------
    mask : array-like of bool or None
        Mask to normalize. If None, returns all True.
    n : int
        Expected length.

    Returns
    -------
    np.ndarray
        Boolean mask of shape (n,).
    """
    if mask is None:
        return np.ones(n, dtype=bool)

    out = np.asarray(mask, dtype=bool).reshape(-1)
    if len(out) != n:
        raise ValueError(f"mask must have length {n}, got {len(out)}.")
    return out


def _extract_data(
    dist: "ParticleDistribution",
    key_or_array: Union[str, ArrayLike],
    *,
    n_expected: Optional[int] = None,
    dtype=None,
    name: str = "data",
) -> np.ndarray:
    """
    Extract a 1D array either from a distribution quantity key or from raw input.

    Parameters
    ----------
    dist : ParticleDistribution
        Source distribution.
    key_or_array : str or array-like
        If str, interpreted as a quantity key in dist.
        Otherwise interpreted as raw data.
    n_expected : int, optional
        Expected output length.
    dtype : optional
        Optional dtype for np.asarray.
    name : str
        Name used in error messages.

    Returns
    -------
    np.ndarray
        1D array.
    """
    if isinstance(key_or_array, str):
        arr = np.asarray(dist.get_data(key_or_array), dtype=dtype).reshape(-1)
    else:
        arr = _as_1d_array(key_or_array, dtype=dtype, name=name)

    if n_expected is not None:
        _validate_length(arr, n_expected, name=name)

    return arr


def _get_weight_array(
    dist: "ParticleDistribution",
    weight: WeightLike = None,
    *,
    absolute: bool = True,
) -> np.ndarray:
    """
    Return a particle weight array.

    Parameters
    ----------
    dist : ParticleDistribution
        Source distribution.
    weight : None, str, or array-like
        - None: uniform weights
        - str: quantity key in dist
        - array-like: explicit particle weights
    absolute : bool
        If True, return absolute value of the weight array.

    Returns
    -------
    np.ndarray
        1D weight array of length len(dist).
    """
    n = len(dist)

    if weight is None:
        w = np.ones(n, dtype=float)
    elif isinstance(weight, str):
        w = np.asarray(dist.get_data(weight), dtype=float).reshape(-1)
    else:
        w = _as_1d_array(weight, dtype=float, name="weight")

    _validate_length(w, n, name="weight")

    if absolute:
        w = np.abs(w)

    return w


def velocity_to_momentum_evc(
    vx: ArrayLike,
    vy: ArrayLike,
    vz: ArrayLike,
    *,
    m0: float = g_m0,
    q: float = g_e0,
    c: float = g_c,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert velocity vector components [m/s] to momentum components [eV/c].

    This helper uses partdist.kinematics for the relativistic magnitude conversion,
    and reconstructs the vector components from the original direction.

    Parameters
    ----------
    vx, vy, vz : array-like
        Velocity components in [m/s].
    m0 : float
        Rest mass in [kg].
    q : float
        Charge in [C].
    c : float
        Speed of light in [m/s].

    Returns
    -------
    px, py, pz : np.ndarray
        Momentum components in [eV/c].
    """
    vx = _as_1d_array(vx, dtype=float, name="vx")
    vy = _as_1d_array(vy, dtype=float, name="vy")
    vz = _as_1d_array(vz, dtype=float, name="vz")

    n = len(vx)
    _validate_length(vy, n, name="vy")
    _validate_length(vz, n, name="vz")

    v_abs = np.sqrt(vx**2 + vy**2 + vz**2)

    if np.any(v_abs >= c):
        raise ValueError("Velocity magnitude must satisfy |v| < c.")

    p_abs = np.asarray(relconv.p_eVc_from_v(v_abs, m0=m0, q=q, c=c), dtype=float)

    px = np.zeros_like(v_abs)
    py = np.zeros_like(v_abs)
    pz = np.zeros_like(v_abs)

    nonzero = v_abs > 0.0
    px[nonzero] = p_abs[nonzero] * vx[nonzero] / v_abs[nonzero]
    py[nonzero] = p_abs[nonzero] * vy[nonzero] / v_abs[nonzero]
    pz[nonzero] = p_abs[nonzero] * vz[nonzero] / v_abs[nonzero]

    return px, py, pz


def momentum_evc_to_velocity(
    px: ArrayLike,
    py: ArrayLike,
    pz: ArrayLike,
    *,
    m0: float = g_m0,
    q: float = g_e0,
    c: float = g_c,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert momentum vector components [eV/c] to velocity components [m/s].

    This helper uses partdist.kinematics for the relativistic magnitude conversion,
    and reconstructs the vector components from the original direction.

    Parameters
    ----------
    px, py, pz : array-like
        Momentum components in [eV/c].
    m0 : float
        Rest mass in [kg].
    q : float
        Charge in [C].
    c : float
        Speed of light in [m/s].

    Returns
    -------
    vx, vy, vz : np.ndarray
        Velocity components in [m/s].
    """
    px = _as_1d_array(px, dtype=float, name="px")
    py = _as_1d_array(py, dtype=float, name="py")
    pz = _as_1d_array(pz, dtype=float, name="pz")

    n = len(px)
    _validate_length(py, n, name="py")
    _validate_length(pz, n, name="pz")

    p_abs = np.sqrt(px**2 + py**2 + pz**2)
    v_abs = np.asarray(relconv.v_from_p_eVc(p_abs, m0=m0, q=q, c=c), dtype=float)

    vx = np.zeros_like(p_abs)
    vy = np.zeros_like(p_abs)
    vz = np.zeros_like(p_abs)

    nonzero = p_abs > 0.0
    vx[nonzero] = v_abs[nonzero] * px[nonzero] / p_abs[nonzero]
    vy[nonzero] = v_abs[nonzero] * py[nonzero] / p_abs[nonzero]
    vz[nonzero] = v_abs[nonzero] * pz[nonzero] / p_abs[nonzero]

    return vx, vy, vz


def _update_momentum_components(
    dist: "ParticleDistribution",
    px: ArrayLike,
    py: ArrayLike,
    pz: ArrayLike,
    *,
    inplace: bool = False,
) -> "ParticleDistribution":
    """
    Update the canonical stored momentum components in a distribution.

    Parameters
    ----------
    dist : ParticleDistribution
        Input distribution.
    px, py, pz : array-like
        New momentum components in [eV/c].
    inplace : bool
        Whether to modify the input object directly.

    Returns
    -------
    ParticleDistribution
        Updated distribution.
    """
    out = _copy_or_inplace(dist, inplace=inplace)
    n = len(out)

    px_arr = _as_1d_array(px, dtype=float, name="px")
    py_arr = _as_1d_array(py, dtype=float, name="py")
    pz_arr = _as_1d_array(pz, dtype=float, name="pz")

    _validate_length(px_arr, n, name="px")
    _validate_length(py_arr, n, name="py")
    _validate_length(pz_arr, n, name="pz")

    out.update_quantity("px", px_arr, inplace=True)
    out.update_quantity("py", py_arr, inplace=True)
    out.update_quantity("pz", pz_arr, inplace=True)

    return out


def _replace_velocity_from_momentum(
    dist: "ParticleDistribution",
    px: ArrayLike,
    py: ArrayLike,
    pz: ArrayLike,
    *,
    m0: float = g_m0,
    q: float = g_e0,
    c: float = g_c,
    inplace: bool = False,
) -> "ParticleDistribution":
    """
    Update the canonical stored momentum components from supplied [eV/c] values.

    Parameters m0, q, c are accepted for API compatibility but not used
    (the values are already in eV/c and stored directly).
    """
    return _update_momentum_components(dist, px, py, pz, inplace=inplace)


def _get_xyz_data(dist: "ParticleDistribution") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return x, y, z as 1D arrays.
    """
    return (
        np.asarray(dist.x, dtype=float).reshape(-1),
        np.asarray(dist.y, dtype=float).reshape(-1),
        np.asarray(dist.z, dtype=float).reshape(-1),
    )


def _get_vxyz_data(dist: "ParticleDistribution") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return vx, vy, vz as 1D arrays.
    These are derived quantities computed from the stored px/py/pz.
    """
    return (
        np.asarray(dist.vx, dtype=float).reshape(-1),
        np.asarray(dist.vy, dtype=float).reshape(-1),
        np.asarray(dist.vz, dtype=float).reshape(-1),
    )


def _get_pxyz_data(
    dist: "ParticleDistribution",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return px, py, pz as 1D arrays.

    These are base quantities stored directly on ParticleDistribution3D
    (see _BASE_SPECS in core.py). The helper exists only to enforce
    1D float dtype at call sites.
    """
    return (
        np.asarray(dist.px, dtype=float).reshape(-1),
        np.asarray(dist.py, dtype=float).reshape(-1),
        np.asarray(dist.pz, dtype=float).reshape(-1),
    )