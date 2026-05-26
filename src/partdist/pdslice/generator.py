"""Parametrised distribution generators for SliceDistribution.

Produces zero-mean, uncorrelated geometric shapes from a small set of
primitives. Twiss matching, centroid shifts, and inter-axis correlations
are out of scope — apply manipulator routines after generation to set
those.

Shapes:
    - Gaussian:       1D normal with optional truncation (ASTRA-style `cut`)
    - Uniform:        1D hard-edged uniform on [mean - L/2, mean + L/2]
    - Plateau:        1D flat top with Fermi-Dirac soft edges of width `r`
    - RadialUniform:  joint 2D uniform on a disk of radius R (KV-like)
    - Isotropic:      joint (px, py, pz) on half-sphere |p|=p_mag, pz >= 0

Units:
    - positions:  metres (SI)
    - momenta:    eV/c  (matches SliceDistribution's stored unit)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np


@dataclass(frozen=True)
class Gaussian:
    """1D Gaussian shape.

    Parameters
    ----------
    sig : float
        rms (= σ) of the input Gaussian (before truncation, if any).
    mean : float
        Centroid. Default 0.0; non-zero only physically meaningful on the
        pz axis (acts as the longitudinal momentum anchor).
    cut : float | None
        Truncation in σ units (ASTRA's ``C_sig_*``). If set, sampling
        draws from a Gaussian restricted to ``[mean - cut*sig, mean + cut*sig]``;
        the *resulting* rms is smaller than ``sig``.
    """
    sig: float
    mean: float = 0.0
    cut: Optional[float] = None

    def __post_init__(self) -> None:
        if self.sig <= 0:
            raise ValueError(f"Gaussian.sig must be > 0, got {self.sig}")
        if self.cut is not None and self.cut <= 0:
            raise ValueError(f"Gaussian.cut must be > 0 if given, got {self.cut}")

    def _sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        if self.cut is None:
            return rng.normal(loc=self.mean, scale=self.sig, size=n)
        from scipy.stats import truncnorm
        a, b = -self.cut, self.cut
        return truncnorm.rvs(
            a, b, loc=self.mean, scale=self.sig, size=n, random_state=rng,
        )


@dataclass(frozen=True)
class Uniform:
    """1D uniform shape on ``[mean - L/2, mean + L/2]``.

    L is FWHM = full support width (hard edges).
    """
    L: float
    mean: float = 0.0

    def __post_init__(self) -> None:
        if self.L <= 0:
            raise ValueError(f"Uniform.L must be > 0, got {self.L}")

    def _sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        half = self.L / 2.0
        return rng.uniform(low=self.mean - half, high=self.mean + half, size=n)


@dataclass(frozen=True)
class Plateau:
    """1D flat-top shape with Fermi-Dirac soft edges.

    PDF (symmetric):
        f(x) ∝ 1 / (1 + exp(2·(2|x - mean| - L) / r))

    L is the full width at half maximum (i.e. f(±L/2) = f(0)/2); the actual
    support extends ~5·r beyond ±L/2 due to the Fermi-Dirac tails.
    """
    L: float
    r: float
    mean: float = 0.0

    def __post_init__(self) -> None:
        if self.L <= 0:
            raise ValueError(f"Plateau.L must be > 0, got {self.L}")
        if self.r <= 0:
            raise ValueError(f"Plateau.r must be > 0, got {self.r}")

    def _sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Rejection-sample from the symmetric Fermi-Dirac plateau.

        f(x) = 1 / (1 + exp(2·(2|x| - L) / r))   (unnormalised, peak = 1)
        """
        half_L = self.L / 2.0
        r = self.r
        window_half = half_L + 5.0 * r
        out = np.empty(n, dtype=float)
        filled = 0
        oversample = max(1.0, 2.5 * window_half / self.L)
        while filled < n:
            trial_n = int((n - filled) * oversample) + 16
            x_try = rng.uniform(-window_half, window_half, size=trial_n)
            u = rng.uniform(0.0, 1.0, size=trial_n)
            f_x = 1.0 / (1.0 + np.exp(2.0 * (2.0 * np.abs(x_try) - self.L) / r))
            accepted = x_try[u <= f_x]
            take = min(n - filled, len(accepted))
            out[filled:filled + take] = accepted[:take]
            filled += take
        return out + self.mean


@dataclass(frozen=True)
class RadialUniform:
    """Joint 2D uniform shape on a disk of radius R centred at origin.

    Used on (x, y) via the ``transverse=`` parameter of ``make_slice``,
    or on (px, py) via ``transverse_momentum=``.
    """
    R: float

    def __post_init__(self) -> None:
        if self.R <= 0:
            raise ValueError(f"RadialUniform.R must be > 0, got {self.R}")

    def _sample2d(self, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Uniform on disk: r = R·√U, θ = 2πU₂ gives uniform area density."""
        u1 = rng.uniform(0.0, 1.0, size=n)
        u2 = rng.uniform(0.0, 1.0, size=n)
        radius = self.R * np.sqrt(u1)
        theta = 2.0 * np.pi * u2
        a = radius * np.cos(theta)
        b = radius * np.sin(theta)
        return a, b


@dataclass(frozen=True)
class Isotropic:
    """Joint (px, py, pz) on a half-sphere of radius p_mag with pz >= 0.

    Physically: isotropic emission into a half-sphere (e.g. cathode emission).
    Implies σ_px = σ_py = p_mag/√3, σ_pz = p_mag/(2√3), <pz> = p_mag/2.

    p_mag is in eV/c.
    """
    p_mag: float

    def __post_init__(self) -> None:
        if self.p_mag <= 0:
            raise ValueError(f"Isotropic.p_mag must be > 0, got {self.p_mag}")

    def _sample3d(self, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Uniform on upper half-sphere |p|=p_mag, pz >= 0.

        cos(θ) uniform on [0, 1] gives uniform area density on the half-sphere.
        """
        cos_th = rng.uniform(0.0, 1.0, size=n)
        sin_th = np.sqrt(1.0 - cos_th ** 2)
        phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
        px = self.p_mag * sin_th * np.cos(phi)
        py = self.p_mag * sin_th * np.sin(phi)
        pz = self.p_mag * cos_th
        return px, py, pz


# Type alias for independent-axis shapes (1D)
_Axis1D = Union[Gaussian, Uniform, Plateau]


def make_slice(
    n: int,
    *,
    I_total: float,
    z: float = 0.0,
    x: Optional[_Axis1D] = None,
    y: Optional[_Axis1D] = None,
    px: Optional[_Axis1D] = None,
    py: Optional[_Axis1D] = None,
    pz: Optional[_Axis1D] = None,
    transverse: Optional[RadialUniform] = None,
    transverse_momentum: Optional[RadialUniform] = None,
    momentum: Optional[Isotropic] = None,
    seed: Optional[int] = None,
):
    """Generate a SliceDistribution from per-axis shape primitives.

    See module docstring for the design rationale and shape semantics.

    Parameters
    ----------
    n : int
        Number of macroparticles (>= 1).
    I_total : float
        Total current in Amperes; uniformly distributed to per-particle
        line charge density ``lam_i = I_total / (<v_z> · n)``.
    z : float
        Longitudinal position of the slice [m].
    x, y, px, py, pz : 1D-shape | None
        Independent-axis shapes. Mutually exclusive with the corresponding
        joint shape.
    transverse : RadialUniform | None
        Joint (x, y) shape. Excludes ``x`` and ``y``.
    transverse_momentum : RadialUniform | None
        Joint (px, py) shape. Excludes ``px`` and ``py``.
    momentum : Isotropic | None
        Joint (px, py, pz) shape. Excludes ``px``, ``py``, ``pz``,
        and ``transverse_momentum``.
    seed : int | None
        Seed for the internal numpy Generator. None ⇒ non-deterministic.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if I_total <= 0:
        raise ValueError(f"I_total must be > 0, got {I_total}")

    if transverse is not None and (x is not None or y is not None):
        raise ValueError(
            "Cannot set 'transverse' together with 'x' or 'y'; "
            "use the joint shape or independent shapes, not both."
        )
    if transverse_momentum is not None and (px is not None or py is not None):
        raise ValueError(
            "Cannot set 'transverse_momentum' together with 'px' or 'py'."
        )
    if momentum is not None and (
        px is not None or py is not None or pz is not None
        or transverse_momentum is not None
    ):
        raise ValueError(
            "Cannot set 'momentum' together with any of 'px', 'py', 'pz', "
            "or 'transverse_momentum'."
        )

    if transverse is None and (x is None or y is None):
        raise ValueError(
            "Must specify either 'transverse' or both 'x' and 'y'."
        )
    if momentum is None:
        if transverse_momentum is None and (px is None or py is None):
            raise ValueError(
                "Must specify 'momentum', 'transverse_momentum', "
                "or both 'px' and 'py'."
            )
        if pz is None:
            raise ValueError("Must specify either 'momentum' or 'pz'.")

    # ---- sampling ----
    rng = np.random.default_rng(seed)

    # In every `else` branch below, the required-fill validation above
    # guarantees the dispatched shape is non-None — mypy cannot follow that
    # control flow, hence the `type: ignore[union-attr]` suppressions.
    if transverse is not None:
        x_arr, y_arr = transverse._sample2d(n, rng)
    else:
        x_arr = x._sample(n, rng)   # type: ignore[union-attr]  # validated non-None above
        y_arr = y._sample(n, rng)   # type: ignore[union-attr]  # validated non-None above

    if momentum is not None:
        px_arr, py_arr, pz_arr = momentum._sample3d(n, rng)
    else:
        if transverse_momentum is not None:
            px_arr, py_arr = transverse_momentum._sample2d(n, rng)
        else:
            px_arr = px._sample(n, rng)  # type: ignore[union-attr]  # validated non-None above
            py_arr = py._sample(n, rng)  # type: ignore[union-attr]  # validated non-None above
        pz_arr = pz._sample(n, rng)      # type: ignore[union-attr]  # validated non-None above

    # ---- I_total → uniform lam ----
    from ..pd3d.utils import momentum_evc_to_velocity
    _vx, _vy, vz = momentum_evc_to_velocity(px_arr, py_arr, pz_arr)
    mean_vz = float(np.mean(vz))
    if mean_vz <= 0:
        raise ValueError(
            f"Mean longitudinal velocity <v_z>={mean_vz} m/s is not positive; "
            f"cannot convert I_total to per-particle lam. Ensure the pz "
            f"shape produces a forward-moving beam (positive mean pz)."
        )
    lam_per_particle = I_total / (mean_vz * n)
    lam_arr = np.full(n, lam_per_particle, dtype=float)

    # All particles emitted simultaneously
    t_arr = np.zeros(n, dtype=float)

    # Local import to avoid a top-level circular dep risk
    from .core import SliceDistribution
    return SliceDistribution(
        z=z,
        x=x_arr, y=y_arr,
        px=px_arr, py=py_arr, pz=pz_arr,
        t=t_arr, lam=lam_arr,
    )
