"""
SliceDistribution — particle distribution on a single transverse slice.

Stores particles whose ``z`` coordinate is fixed to a scalar slice
position ``z0``. The transverse columns ``(x, y, px, py, pz, t)`` and the
linear charge density ``lam`` [C/m] are per-particle arrays. Derived
quantities (gamma, beta, vx, vy, vz, xp, current, …) are computed lazily
from the base columns and cached where the cost matters.

Designed as a sibling of :class:`partdist.ParticleDistribution3D`; same
derived-quantity machinery, same Twiss diagnostics via
:func:`partdist.pd3d.analysis.compute_twiss_plane`. CST Particle Studio
``.pid`` emission files are the canonical input format
(:func:`partdist.read_cst_pid_distribution`).
"""
from __future__ import annotations

import pandas as pd
from typing import Dict, Mapping, Sequence, Union

import numpy as np
from scipy.constants import c as g_c, m_e as g_m0, e as g_e0
from partdist import kinematics as relconv

from ..particle_array_quantity import (
    ParticleArrayQuantity,
    QuantityCategory,
    QuantityDTypeKind,
)
from ..pd3d.analysis import compute_twiss_plane

from .._array_helpers import as_1d_float_array as _as_1d_float_array

ArrayLike = Union[float, Sequence[float], np.ndarray]


def _make_paq(key: str, data: np.ndarray, spec: dict) -> ParticleArrayQuantity:
    return ParticleArrayQuantity(
        name=key,
        data=data,
        unit=spec["unit"],
        dtype_kind=spec["dtype_kind"],
        short_name=spec["short_name"],
        long_name=spec["long_name"],
        latex_name=spec["latex_name"],
        category=spec["category"],
        is_derived=spec["is_derived"],
    )


class SliceDistribution:
    """
    2.5D particle distribution on a transverse slice perpendicular to the
    z-axis.  All particles share a common ``z`` position; ``x`` and ``y``
    vary.  Full 3-component momenta (px, py, pz) are stored per-particle
    [eV/c].

    z is the longitudinal / propagation direction by convention — this
    matches accelerator-physics usage and is consistent with
    :class:`partdist.ParticleDistribution3D`.  The class is intentionally
    z-only; if you need a slice along x or y, rotate your data first.

    The charge quantity ``lam`` (λ) is the *linear* charge density [C/m],
    i.e. charge per unit length along z.

    Accessing ``.z`` returns a Python ``float`` (the slice's z₀) — not an
    ndarray — to make the distinction from a 3D distribution immediately
    visible and to prevent accidental mixing of the two types.

    Parameters
    ----------
    z : float
        The common z-coordinate of the slice [m].
    x, y : ArrayLike
        Per-particle transverse positions [m].
    px, py, pz : ArrayLike
        Momentum components [eV/c].
    t : ArrayLike
        Time [s].
    lam : ArrayLike
        Linear charge density λ [C/m].
    extras : mapping, optional
        Additional per-particle quantities.
    """

    _POSITION_SPECS: dict[str, dict] = {
        "x": {"unit": "m",   "dtype_kind": "float", "short_name": "x", "long_name": "horizontal position",   "latex_name": r"$x$",   "category": "position", "is_derived": False},
        "y": {"unit": "m",   "dtype_kind": "float", "short_name": "y", "long_name": "vertical position",     "latex_name": r"$y$",   "category": "position", "is_derived": False},
        "z": {"unit": "m",   "dtype_kind": "float", "short_name": "z", "long_name": "longitudinal position", "latex_name": r"$z$",   "category": "position", "is_derived": False},
    }

    _NON_POSITION_BASE_SPECS: dict[str, dict] = {
        "px": {"unit": "eV/c", "dtype_kind": "float", "short_name": "px", "long_name": "horizontal momentum",   "latex_name": r"$p_x$",      "category": "momentum", "is_derived": False},
        "py": {"unit": "eV/c", "dtype_kind": "float", "short_name": "py", "long_name": "vertical momentum",     "latex_name": r"$p_y$",      "category": "momentum", "is_derived": False},
        "pz": {"unit": "eV/c", "dtype_kind": "float", "short_name": "pz", "long_name": "longitudinal momentum", "latex_name": r"$p_z$",      "category": "momentum", "is_derived": False},
        "t":  {"unit": "s",    "dtype_kind": "float", "short_name": "t",  "long_name": "time",                  "latex_name": r"$t$",        "category": "time",     "is_derived": False},
        "lam": {"unit": "C/m",  "dtype_kind": "float", "short_name": "λ",   "long_name": "linear charge density", "latex_name": r"$\lambda$",   "category": "charge",  "is_derived": False},
    }

    _DERIVED_SPECS: dict[str, dict] = {
        "lam_abs":            {"unit": "C/m",    "dtype_kind": "float", "short_name": "|λ|",        "long_name": "absolute linear charge density", "latex_name": r"$|\lambda|$",     "category": "charge",   "is_derived": True},
        "vx":                 {"unit": "m/s",    "dtype_kind": "float", "short_name": "vx",         "long_name": "horizontal velocity",            "latex_name": r"$v_x$",           "category": "velocity", "is_derived": True},
        "vy":                 {"unit": "m/s",    "dtype_kind": "float", "short_name": "vy",         "long_name": "vertical velocity",              "latex_name": r"$v_y$",           "category": "velocity", "is_derived": True},
        "vz":                 {"unit": "m/s",    "dtype_kind": "float", "short_name": "vz",         "long_name": "longitudinal velocity",          "latex_name": r"$v_z$",           "category": "velocity", "is_derived": True},
        "radial_position":    {"unit": "m",      "dtype_kind": "float", "short_name": "r",          "long_name": "radial position",                "latex_name": r"$r$",             "category": "geometry", "is_derived": True},
        "transverse_speed":   {"unit": "m/s",    "dtype_kind": "float", "short_name": "v_perp",     "long_name": "transverse speed",               "latex_name": r"$v_{\perp}$",     "category": "geometry", "is_derived": True},
        "radial_velocity":    {"unit": "m/s",    "dtype_kind": "float", "short_name": "v_r",        "long_name": "radial velocity",                "latex_name": r"$v_r$",           "category": "geometry", "is_derived": True},
        "azimuthal_velocity": {"unit": "m/s",    "dtype_kind": "float", "short_name": "v_phi",      "long_name": "azimuthal velocity",             "latex_name": r"$v_{\phi}$",      "category": "geometry", "is_derived": True},
        "speed":              {"unit": "m/s",    "dtype_kind": "float", "short_name": "v",          "long_name": "speed",                          "latex_name": r"$v$",             "category": "velocity", "is_derived": True},
        "beta":               {"unit": "",       "dtype_kind": "float", "short_name": "beta",       "long_name": "normalized speed",               "latex_name": r"$\beta$",         "category": "velocity", "is_derived": True},
        "gamma":              {"unit": "",       "dtype_kind": "float", "short_name": "gamma",      "long_name": "Lorentz factor",                 "latex_name": r"$\gamma$",        "category": "velocity", "is_derived": True},
        "xp":                 {"unit": "rad",    "dtype_kind": "float", "short_name": "xp",         "long_name": "horizontal normalized angle",    "latex_name": r"$x'$",            "category": "other",    "is_derived": True},
        "yp":                 {"unit": "rad",    "dtype_kind": "float", "short_name": "yp",         "long_name": "vertical normalized angle",      "latex_name": r"$y'$",            "category": "other",    "is_derived": True},
        "p_abs_si":           {"unit": "kg*m/s", "dtype_kind": "float", "short_name": "p",          "long_name": "momentum magnitude (SI)",        "latex_name": r"$|p|$",           "category": "momentum", "is_derived": True},
        "p_abs":              {"unit": "eV/c",   "dtype_kind": "float", "short_name": "p",          "long_name": "momentum magnitude",             "latex_name": r"$|p|$",           "category": "momentum", "is_derived": True},
        "kinetic_energy":     {"unit": "J",      "dtype_kind": "float", "short_name": "Ek",         "long_name": "kinetic energy",                 "latex_name": r"$E_k$",           "category": "energy",   "is_derived": True},
        "kinetic_energy_eV":  {"unit": "eV",     "dtype_kind": "float", "short_name": "Ek",         "long_name": "kinetic energy",                 "latex_name": r"$E_k$",           "category": "energy",   "is_derived": True},
        "current":     {"unit": "A",  "dtype_kind": "float", "short_name": "I", "long_name": "beam current through the slice", "latex_name": r"$I$", "category": "current", "is_derived": True},
        "current_abs": {"unit": "A",  "dtype_kind": "float", "short_name": "|I|", "long_name": "absolute beam current through the slice", "latex_name": r"$|I|$", "category": "current", "is_derived": True},
    }

    # ------------------------------------------------------------------ #
    # Construction                                                         #
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        *,
        z: float,
        x: ArrayLike,
        y: ArrayLike,
        px: ArrayLike,
        py: ArrayLike,
        pz: ArrayLike,
        t: ArrayLike,
        lam: ArrayLike,
        extras: Mapping[str, ArrayLike] | Mapping[str, ParticleArrayQuantity] | None = None,
    ) -> None:
        self._z0: float = float(z)
        self._quantities: Dict[str, ParticleArrayQuantity] = {}
        self._velocity_cache: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

        for axis, arr_in in (("x", x), ("y", y)):
            arr = _as_1d_float_array(arr_in, axis)
            self._quantities[axis] = _make_paq(axis, arr, self._POSITION_SPECS[axis])

        n = len(self._quantities["x"].data)
        if len(self._quantities["y"].data) != n:
            raise ValueError("x and y must have the same length.")

        non_pos = {
            "px": _as_1d_float_array(px, "px"),
            "py": _as_1d_float_array(py, "py"),
            "pz": _as_1d_float_array(pz, "pz"),
            "t":  _as_1d_float_array(t,  "t"),
            "lam": _as_1d_float_array(lam, "lam"),
        }
        for key, arr in non_pos.items():
            if len(arr) != n:
                raise ValueError(f"All arrays must have the same length; '{key}' mismatches.")
            self._quantities[key] = _make_paq(key, arr, self._NON_POSITION_BASE_SPECS[key])

        if extras is not None:
            for key, value in extras.items():
                self.add_quantity(key, value, inplace=True)

    def __len__(self) -> int:
        return self.n

    def __repr__(self) -> str:
        return f"SliceDistribution(z={self._z0:.6g} m, n={self.n})"

    @classmethod
    def from_arrays(
        cls,
        *,
        z: float,
        x: ArrayLike,
        y: ArrayLike,
        px: ArrayLike,
        py: ArrayLike,
        pz: ArrayLike,
        t: ArrayLike,
        lam: ArrayLike | None = None,
        extras: Mapping[str, ArrayLike] | Mapping[str, ParticleArrayQuantity] | None = None,
    ) -> "SliceDistribution":
        if lam is None:
            lam = np.ones(len(np.asarray(x, dtype=float).reshape(-1)), dtype=float)
        return cls(
            z=z, x=x, y=y, px=px, py=py, pz=pz, t=t, lam=lam, extras=extras,
        )

    @classmethod
    def from_dict(cls, data: Mapping) -> "SliceDistribution":
        required = {"z", "x", "y", "px", "py", "pz", "t", "lam"}
        missing = [k for k in required if k not in data]
        if missing:
            raise KeyError(f"Missing required keys: {missing}")
        extras = {k: v for k, v in data.items() if k not in required}
        return cls(
            z=data["z"], x=data["x"], y=data["y"],
            px=data["px"], py=data["py"], pz=data["pz"],
            t=data["t"], lam=data["lam"],
            extras=extras or None,
        )

    # ------------------------------------------------------------------ #
    # Axis / key metadata                                                  #
    # ------------------------------------------------------------------ #

    @property
    def _stored_base_keys(self) -> tuple[str, ...]:
        return ("x", "y", "px", "py", "pz", "t", "lam")

    @property
    def base_quantity_keys(self) -> tuple[str, ...]:
        return self._stored_base_keys

    @property
    def derived_quantity_keys(self) -> tuple[str, ...]:
        return tuple(self._DERIVED_SPECS.keys())

    @property
    def extra_quantity_keys(self) -> tuple[str, ...]:
        all_builtin = set(self._stored_base_keys) | set(self._DERIVED_SPECS)
        return tuple(k for k in self._quantities if k not in all_builtin)

    @property
    def quantity_keys(self) -> tuple[str, ...]:
        return self.base_quantity_keys + self.derived_quantity_keys + self.extra_quantity_keys

    def has_quantity(self, key: str) -> bool:
        return key in self.quantity_keys or key == "z"

    # ------------------------------------------------------------------ #
    # Size                                                                 #
    # ------------------------------------------------------------------ #

    @property
    def n(self) -> int:
        return self._quantities["x"].n

    @property
    def size(self) -> int:
        return self.n

    # ------------------------------------------------------------------ #
    # Charge summary                                                       #
    # ------------------------------------------------------------------ #

    @property
    def total_linear_charge(self) -> float:
        """Sum of lam [C/m]. Multiply by a length along the fixed axis for total charge."""
        return float(np.sum(self.lam))

    @property
    def total_linear_charge_abs(self) -> float:
        return float(np.sum(np.abs(self.lam)))

    # ------------------------------------------------------------------ #
    # Quantity access                                                      #
    # ------------------------------------------------------------------ #

    def _make_derived_quantity(self, key: str) -> ParticleArrayQuantity:
        spec = self._DERIVED_SPECS[key]
        data = getattr(self, f"_calc_{key}")()
        return _make_paq(key, data, spec)

    def get_quantity(self, key: str) -> ParticleArrayQuantity:
        if key == "z":
            raise ValueError(
                "'z' is the fixed axis of a SliceDistribution. "
                "Access its scalar value via .z (returns float)."
            )
        if key in self._quantities:
            return self._quantities[key]
        if key in self._DERIVED_SPECS:
            return self._make_derived_quantity(key)
        raise KeyError(f"Unknown quantity: {key!r}")

    def get_data(self, key: str) -> np.ndarray:
        return self.get_quantity(key).data

    def get(self, key: str) -> np.ndarray:
        return self.get_data(key)

    def get_quantity_unit(self, key: str) -> str:
        return self.get_quantity(key).unit

    def get_quantity_category(self, key: str) -> str:
        return self.get_quantity(key).category

    def get_quantity_short_name(self, key: str) -> str:
        return self.get_quantity(key).short_name

    def get_quantity_long_name(self, key: str) -> str:
        return self.get_quantity(key).long_name

    def get_quantity_latex_name(self, key: str) -> str:
        return self.get_quantity(key).latex_name

    # ------------------------------------------------------------------ #
    # Quantity management                                                  #
    # ------------------------------------------------------------------ #

    def _all_builtin_keys(self) -> set[str]:
        return (
            set(self._POSITION_SPECS)
            | set(self._NON_POSITION_BASE_SPECS)
            | set(self._DERIVED_SPECS)
        )

    def add_quantity(
        self,
        key: str,
        value: ArrayLike | ParticleArrayQuantity,
        *,
        unit: str = "",
        dtype_kind: QuantityDTypeKind = "float",
        short_name: str | None = None,
        long_name: str | None = None,
        latex_name: str | None = None,
        category: QuantityCategory = "other",
        is_derived: bool = False,
        is_discrete: bool | None = None,
        preferred_scale: str = "linear",
        inplace: bool = True,
    ) -> "SliceDistribution":
        if key in self._all_builtin_keys():
            raise ValueError(f"'{key}' conflicts with a built-in quantity name.")
        if key in self._quantities:
            raise ValueError(f"Quantity '{key}' already exists. Use update_quantity(...) instead.")

        out = self if inplace else self.copy()

        if isinstance(value, ParticleArrayQuantity):
            q = value.copy()
            if q.name != key:
                q.name = key
        else:
            arr = _as_1d_float_array(value, key)
            q = ParticleArrayQuantity(
                name=key, data=arr, unit=unit, dtype_kind=dtype_kind,
                short_name=short_name, long_name=long_name, latex_name=latex_name,
                category=category, is_derived=is_derived,
                is_discrete=is_discrete, preferred_scale=preferred_scale,
            )

        if q.n != out.n:
            raise ValueError(f"Quantity '{key}' must have length {out.n}, got {q.n}.")

        out._quantities[key] = q
        return out

    def update_quantity(
        self,
        key: str,
        value: ArrayLike | ParticleArrayQuantity,
        *,
        update_meta: bool = False,
        inplace: bool = True,
    ) -> "SliceDistribution":
        if key in self._DERIVED_SPECS:
            raise ValueError(
                f"Cannot update derived quantity '{key}' directly. "
                "Update the underlying base quantities instead."
            )
        if key == "z":
            raise ValueError(
                "'z' is the fixed axis of a SliceDistribution. "
                "To change its value, set .z directly."
            )
        if key not in self._quantities:
            raise KeyError(f"Quantity '{key}' does not exist. Use add_quantity(...) instead.")

        out = self if inplace else self.copy()
        old_q = out._quantities[key]

        if isinstance(value, ParticleArrayQuantity):
            new_q = value.copy()
            if new_q.name != key:
                new_q.name = key
        else:
            arr = _as_1d_float_array(value, key)
            new_q = ParticleArrayQuantity(
                name=key, data=arr, unit=old_q.unit, dtype_kind=old_q.dtype_kind,
                short_name=old_q.short_name, long_name=old_q.long_name,
                latex_name=old_q.latex_name, category=old_q.category,
                is_derived=old_q.is_derived, is_discrete=old_q.is_discrete,
                preferred_scale=old_q.preferred_scale,
            )

        if new_q.n != out.n:
            raise ValueError(f"Quantity '{key}' must have length {out.n}, got {new_q.n}.")

        if key in set(out._stored_base_keys):
            # Base quantities: data only, metadata is immutable
            out._quantities[key].data = np.asarray(new_q.data, dtype=float).reshape(-1)
            out._velocity_cache = None
        elif update_meta and isinstance(value, ParticleArrayQuantity):
            out._quantities[key] = new_q
        else:
            out._quantities[key].data = new_q.data

        return out

    def drop_quantity(self, key: str, *, inplace: bool = True) -> "SliceDistribution":
        if key in set(self._stored_base_keys):
            raise ValueError(f"Cannot drop base quantity '{key}'.")
        if key in self._DERIVED_SPECS:
            raise ValueError(f"Derived quantity '{key}' is not stored and cannot be dropped.")
        if key not in self._quantities:
            raise KeyError(f"Quantity '{key}' not found.")
        out = self if inplace else self.copy()
        del out._quantities[key]
        return out

    def add_extra(
        self,
        key: str,
        value: ArrayLike | ParticleArrayQuantity,
        *,
        unit: str = "",
        dtype_kind: QuantityDTypeKind = "float",
        short_name: str | None = None,
        long_name: str | None = None,
        latex_name: str | None = None,
        category: QuantityCategory = "other",
        is_discrete: bool | None = None,
        preferred_scale: str = "linear",
        inplace: bool = True,
    ) -> "SliceDistribution":
        if key in self._all_builtin_keys():
            raise ValueError(f"'{key}' conflicts with a built-in quantity name.")
        if key in self._quantities:
            raise ValueError(f"Extra quantity '{key}' already exists. Use update_extra(...) instead.")
        return self.add_quantity(
            key, value, unit=unit, dtype_kind=dtype_kind,
            short_name=short_name, long_name=long_name, latex_name=latex_name,
            category=category, is_derived=False, is_discrete=is_discrete,
            preferred_scale=preferred_scale, inplace=inplace,
        )

    def update_extra(
        self,
        key: str,
        value: ArrayLike | ParticleArrayQuantity,
        *,
        update_meta: bool = False,
        inplace: bool = True,
    ) -> "SliceDistribution":
        if key in set(self._stored_base_keys):
            raise ValueError(f"'{key}' is a base quantity, not an extra quantity.")
        if key in self._DERIVED_SPECS:
            raise ValueError(f"'{key}' is a derived quantity, not an extra quantity.")
        if key not in self._quantities:
            raise KeyError(f"Extra quantity '{key}' does not exist. Use add_extra(...) instead.")
        return self.update_quantity(key, value, update_meta=update_meta, inplace=inplace)

    def drop_extra(self, key: str, *, inplace: bool = True) -> "SliceDistribution":
        if key in set(self._stored_base_keys):
            raise ValueError(f"'{key}' is a base quantity, not an extra quantity.")
        if key in self._DERIVED_SPECS:
            raise ValueError(f"'{key}' is a derived quantity.")
        if key not in self._quantities:
            raise KeyError(f"Extra quantity '{key}' not found.")
        return self.drop_quantity(key, inplace=inplace)

    # ------------------------------------------------------------------ #
    # copy / slice / sort                                                  #
    # ------------------------------------------------------------------ #

    def copy(self) -> "SliceDistribution":
        stored_base = set(self._stored_base_keys)
        extras = {k: q.copy() for k, q in self._quantities.items() if k not in stored_base}
        return SliceDistribution(
            z=self._z0,
            x=self._quantities["x"].data.copy(),
            y=self._quantities["y"].data.copy(),
            px=self.px.copy(), py=self.py.copy(), pz=self.pz.copy(),
            t=self.t.copy(), lam=self.lam.copy(),
            extras=extras or None,
        )

    def slice(self, mask) -> "SliceDistribution":
        extras = {}
        for key in self.extra_quantity_keys:
            q = self.get_quantity(key)
            extras[key] = ParticleArrayQuantity(
                name=q.name, data=q.data[mask], unit=q.unit, dtype_kind=q.dtype_kind,
                short_name=q.short_name, long_name=q.long_name, latex_name=q.latex_name,
                category=q.category, is_derived=q.is_derived,
                is_discrete=q.is_discrete, preferred_scale=q.preferred_scale,
            )
        return SliceDistribution(
            z=self._z0,
            x=self._quantities["x"].data[mask],
            y=self._quantities["y"].data[mask],
            px=self.px[mask], py=self.py[mask], pz=self.pz[mask],
            t=self.t[mask], lam=self.lam[mask],
            extras=extras or None,
        )

    def sort_by(self, key: str, *, ascending: bool = True) -> "SliceDistribution":
        order = np.argsort(self.get_data(key))
        if not ascending:
            order = order[::-1]
        return self.slice(order)

    def update_data(
        self, key: str, data: ArrayLike, *, inplace: bool = False
    ) -> "SliceDistribution":
        return self.update_quantity(key, data, update_meta=False, inplace=inplace)

    # ------------------------------------------------------------------ #
    # Statistics                                                           #
    # ------------------------------------------------------------------ #

    def _get_weights(self, weight: str | np.ndarray | None = "abslam") -> np.ndarray:
        if weight is None:
            w = np.ones(self.size, dtype=float)
        elif isinstance(weight, str):
            if weight == "lam":
                w = np.asarray(self.lam, dtype=float)
            elif weight == "abslam":
                w = np.abs(np.asarray(self.lam, dtype=float))
            else:
                w = np.asarray(self.get_data(weight), dtype=float)
        else:
            w = np.asarray(weight, dtype=float)
        if w.ndim != 1 or len(w) != self.size:
            raise ValueError(f"Weights must be a 1D array of length {self.size}.")
        return w

    def mean(self, key: str, weight: str | np.ndarray | None = "abslam") -> float:
        x = np.asarray(self.get_data(key), dtype=float)
        w = self._get_weights(weight)
        wsum = np.sum(w)
        if wsum == 0.0:
            raise ValueError("Sum of weights is zero.")
        return float(np.sum(w * x) / wsum)

    def var(self, key: str, weight: str | np.ndarray | None = "abslam") -> float:
        x = np.asarray(self.get_data(key), dtype=float)
        w = self._get_weights(weight)
        mu = self.mean(key, weight=weight)
        wsum = np.sum(w)
        if wsum == 0.0:
            raise ValueError("Sum of weights is zero.")
        return float(np.sum(w * (x - mu) ** 2) / wsum)

    def std(self, key: str, weight: str | np.ndarray | None = "abslam") -> float:
        return float(np.sqrt(self.var(key, weight=weight)))

    def rms(self, key: str, weight: str | np.ndarray | None = "abslam") -> float:
        x = np.asarray(self.get_data(key), dtype=float)
        w = self._get_weights(weight)
        wsum = np.sum(w)
        if wsum == 0.0:
            raise ValueError("Sum of weights is zero.")
        return float(np.sqrt(np.sum(w * x**2) / wsum))

    def covariance(
        self, key1: str, key2: str, weight: str | np.ndarray | None = "abslam"
    ) -> float:
        x1 = np.asarray(self.get_data(key1), dtype=float)
        x2 = np.asarray(self.get_data(key2), dtype=float)
        w = self._get_weights(weight)
        mu1 = self.mean(key1, weight=weight)
        mu2 = self.mean(key2, weight=weight)
        wsum = np.sum(w)
        if wsum == 0.0:
            raise ValueError("Sum of weights is zero.")
        return float(np.sum(w * (x1 - mu1) * (x2 - mu2)) / wsum)

    def correlation(
        self, key1: str, key2: str, weight: str | np.ndarray | None = "abslam"
    ) -> float:
        s1 = self.std(key1, weight=weight)
        s2 = self.std(key2, weight=weight)
        if s1 == 0.0 or s2 == 0.0:
            return float("nan")
        return float(self.covariance(key1, key2, weight=weight) / (s1 * s2))

    def linear_fit(
        self, xkey: str, ykey: str, weight: str | np.ndarray | None = "abslam"
    ) -> tuple[float, float]:
        x = np.asarray(self.get_data(xkey), dtype=float)
        y = np.asarray(self.get_data(ykey), dtype=float)
        w = self._get_weights(weight)
        x_mean = self.mean(xkey, weight=weight)
        y_mean = self.mean(ykey, weight=weight)
        dx = x - x_mean
        var_x = np.sum(w * dx**2)
        if var_x == 0.0:
            raise ValueError(f"Cannot fit {ykey} vs {xkey}: zero variance in {xkey}.")
        slope = float(np.sum(w * dx * (y - y_mean)) / var_x)
        intercept = y_mean - slope * x_mean
        return slope, intercept

    def centroid(self, weight: str | np.ndarray | None = "abslam") -> dict[str, float]:
        """Centroid of all base quantities.  ``z`` returns the slice's scalar z₀."""
        result: dict[str, float] = {"z": self._z0}
        for key in ("x", "y", "px", "py", "pz", "t", "lam"):
            result[key] = self.mean(key, weight=weight)
        return result

    def sigma_dict(self, weight: str | np.ndarray | None = "abslam") -> dict[str, float]:
        """RMS widths of all base quantities.  ``z`` returns 0."""
        result: dict[str, float] = {"z": 0.0}
        for key in ("x", "y", "px", "py", "pz", "t", "lam"):
            result[key] = self.std(key, weight=weight)
        return result

    # ------------------------------------------------------------------ #
    # Derived quantity calculations (identical physics to 3D)              #
    # ------------------------------------------------------------------ #

    def _calc_p_abs(self) -> np.ndarray:
        return np.sqrt(
            self._quantities["px"].data ** 2
            + self._quantities["py"].data ** 2
            + self._quantities["pz"].data ** 2
        )

    def _calc_p_abs_si(self) -> np.ndarray:
        return self.p_abs * (abs(g_e0) / g_c)

    def _calc_gamma(self) -> np.ndarray:
        p_si = self.p_abs_si
        return np.sqrt(1.0 + (p_si / (g_m0 * g_c)) ** 2)

    def _calc_beta(self) -> np.ndarray:
        g = self.gamma
        return np.sqrt(1.0 - 1.0 / g ** 2)

    def _calc_speed(self) -> np.ndarray:
        return self.beta * g_c

    def _calc_velocities(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._velocity_cache is None:
            from ..pd3d.utils import momentum_evc_to_velocity
            self._velocity_cache = momentum_evc_to_velocity(
                self._quantities["px"].data,
                self._quantities["py"].data,
                self._quantities["pz"].data,
            )
        return self._velocity_cache

    def _calc_vx(self) -> np.ndarray:
        return self._calc_velocities()[0]

    def _calc_vy(self) -> np.ndarray:
        return self._calc_velocities()[1]

    def _calc_vz(self) -> np.ndarray:
        return self._calc_velocities()[2]

    def _calc_radial_position(self) -> np.ndarray:
        x = self._quantities["x"].data
        y = self._quantities["y"].data
        return np.sqrt(x ** 2 + y ** 2)

    def _calc_transverse_speed(self) -> np.ndarray:
        vx, vy, _ = self._calc_velocities()
        return np.sqrt(vx ** 2 + vy ** 2)

    def _calc_radial_velocity(self) -> np.ndarray:
        x = self._quantities["x"].data
        y = self._quantities["y"].data
        vx, vy, _ = self._calc_velocities()
        r = np.sqrt(x ** 2 + y ** 2) + 1e-30
        return (x * vx + y * vy) / r

    def _calc_azimuthal_velocity(self) -> np.ndarray:
        x = self._quantities["x"].data
        y = self._quantities["y"].data
        vx, vy, _ = self._calc_velocities()
        r = np.sqrt(x ** 2 + y ** 2) + 1e-30
        return (x * vy - y * vx) / r

    def _calc_xp(self) -> np.ndarray:
        return self._quantities["px"].data / self._quantities["pz"].data

    def _calc_yp(self) -> np.ndarray:
        return self._quantities["py"].data / self._quantities["pz"].data

    def _calc_lam_abs(self) -> np.ndarray:
        return np.abs(self.lam)

    def _calc_kinetic_energy(self) -> np.ndarray:
        return (self.gamma - 1.0) * (g_m0 * g_c ** 2)

    def _calc_kinetic_energy_eV(self) -> np.ndarray:
        return self.kinetic_energy / abs(g_e0)

    def _calc_current(self) -> np.ndarray:
        # lam [C/m] * vz [m/s] = I [A] through the slice cross-section.
        return self.lam * self._calc_vz()

    def _calc_current_abs(self) -> np.ndarray:
        return np.abs(self.current)

    def momentum_si(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        factor = abs(g_e0) / g_c
        return (
            self._quantities["px"].data * factor,
            self._quantities["py"].data * factor,
            self._quantities["pz"].data * factor,
        )

    def momentum_evc(
        self,
        *,
        copy: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (px, py, pz) in eV/c.

        With ``copy=True`` (default) the caller gets independent arrays that
        can be mutated freely; the slice's internal state is protected.
        With ``copy=False`` the caller receives the underlying ``_quantities``
        arrays directly — zero allocation, but mutating them mutates the
        slice. Use the zero-copy path only when the caller is read-only and
        N is large enough that the copy cost matters.
        """
        px_data = self._quantities["px"].data
        py_data = self._quantities["py"].data
        pz_data = self._quantities["pz"].data
        if copy:
            return px_data.copy(), py_data.copy(), pz_data.copy()
        return px_data, py_data, pz_data

    # ------------------------------------------------------------------ #
    # Beam-physics shortcuts                                               #
    # ------------------------------------------------------------------ #

    @property
    def gamma0(self) -> float:
        return float(relconv.gamma_from_ke_eV(self.mean("kinetic_energy_eV")))

    @property
    def beta0(self) -> float:
        return float(relconv.beta_from_ke_eV(self.mean("kinetic_energy_eV")))

    @property
    def emit_x(self) -> float:
        """Geometric emittance in x [m·rad]."""
        var_x = self.var("x")
        var_xp = self.var("xp")
        cov = self.covariance("x", "xp")
        return float(np.sqrt(max(var_x * var_xp - cov ** 2, 0.0)))

    @property
    def emit_y(self) -> float:
        """Geometric emittance in y [m·rad]."""
        var_y = self.var("y")
        var_yp = self.var("yp")
        cov = self.covariance("y", "yp")
        return float(np.sqrt(max(var_y * var_yp - cov ** 2, 0.0)))

    @property
    def nemit_x(self) -> float:
        return float(self.beta0 * self.gamma0 * self.emit_x)

    @property
    def nemit_y(self) -> float:
        return float(self.beta0 * self.gamma0 * self.emit_y)

    @property
    def alpha_x(self) -> float:
        """Twiss alpha in x."""
        return compute_twiss_plane(self, plane="x", weight="lam_abs").alpha

    @property
    def beta_x(self) -> float:
        """Twiss beta in x [m]."""
        return compute_twiss_plane(self, plane="x", weight="lam_abs").beta

    @property
    def gamma_x(self) -> float:
        """Twiss gamma in x [1/m]."""
        return compute_twiss_plane(self, plane="x", weight="lam_abs").gamma_twiss

    @property
    def alpha_y(self) -> float:
        """Twiss alpha in y."""
        return compute_twiss_plane(self, plane="y", weight="lam_abs").alpha

    @property
    def beta_y(self) -> float:
        """Twiss beta in y [m]."""
        return compute_twiss_plane(self, plane="y", weight="lam_abs").beta

    @property
    def gamma_y(self) -> float:
        """Twiss gamma in y [1/m]."""
        return compute_twiss_plane(self, plane="y", weight="lam_abs").gamma_twiss

    @property
    def twiss(self) -> Dict[str, float]:
        return dict(
            alpha_x=self.alpha_x,
            beta_x=self.beta_x,
            alpha_y=self.alpha_y,
            beta_y=self.beta_y,
        )

    # ------------------------------------------------------------------ #
    # Convenience properties                                               #
    # ------------------------------------------------------------------ #

    @property
    def x(self) -> np.ndarray:
        """Horizontal position [m]."""
        return self._quantities["x"].data

    @property
    def y(self) -> np.ndarray:
        """Vertical position [m]."""
        return self._quantities["y"].data

    @property
    def z(self) -> float:
        """Longitudinal slice position z₀ [m] — scalar by construction."""
        return self._z0

    @z.setter
    def z(self, value: float) -> None:
        self._z0 = float(value)

    @property
    def px(self) -> np.ndarray:
        return self._quantities["px"].data

    @property
    def py(self) -> np.ndarray:
        return self._quantities["py"].data

    @property
    def pz(self) -> np.ndarray:
        return self._quantities["pz"].data

    @property
    def vx(self) -> np.ndarray:
        return self._calc_vx()

    @property
    def vy(self) -> np.ndarray:
        return self._calc_vy()

    @property
    def vz(self) -> np.ndarray:
        return self._calc_vz()

    @property
    def t(self) -> np.ndarray:
        return self._quantities["t"].data

    @property
    def lam(self) -> np.ndarray:
        """Linear charge density λ [C/m]."""
        return self._quantities["lam"].data

    @property
    def lam_abs(self) -> np.ndarray:
        """Absolute linear charge density |λ| [C/m]."""
        return self._calc_lam_abs()

    @property
    def radial_position(self) -> np.ndarray:
        return self._calc_radial_position()

    @property
    def transverse_speed(self) -> np.ndarray:
        return self._calc_transverse_speed()

    @property
    def radial_velocity(self) -> np.ndarray:
        return self._calc_radial_velocity()

    @property
    def azimuthal_velocity(self) -> np.ndarray:
        return self._calc_azimuthal_velocity()

    @property
    def speed(self) -> np.ndarray:
        return self._calc_speed()

    @property
    def beta(self) -> np.ndarray:
        return self._calc_beta()

    @property
    def gamma(self) -> np.ndarray:
        return self._calc_gamma()

    @property
    def xp(self) -> np.ndarray:
        return self._calc_xp()

    @property
    def yp(self) -> np.ndarray:
        return self._calc_yp()

    @property
    def p_abs_si(self) -> np.ndarray:
        return self._calc_p_abs_si()

    @property
    def p_abs(self) -> np.ndarray:
        return self._calc_p_abs()

    @property
    def kinetic_energy(self) -> np.ndarray:
        return self._calc_kinetic_energy()

    @property
    def kinetic_energy_eV(self) -> np.ndarray:
        return self._calc_kinetic_energy_eV()

    @property
    def current(self) -> np.ndarray:
        """Beam current through the slice cross-section [A] = lam * vz."""
        return self._calc_current()

    @property
    def current_abs(self) -> np.ndarray:
        """Absolute beam current [A]."""
        return self._calc_current_abs()

    @property
    def extras(self) -> Dict[str, ParticleArrayQuantity]:
        return {k: self._quantities[k] for k in self.extra_quantity_keys}

    # ------------------------------------------------------------------ #
    # Serialisation                                                        #
    # ------------------------------------------------------------------ #

    def to_dict(
        self,
        *,
        include_extras: bool = True,
        include_derived: bool = False,
        copy: bool = True,
    ) -> dict:
        out: dict = {"z": self._z0}
        for key in self._stored_base_keys:
            arr = self.get_data(key)
            out[key] = arr.copy() if copy else arr
        if include_extras:
            for key in self.extra_quantity_keys:
                arr = self.get_data(key)
                out[key] = arr.copy() if copy else arr
        if include_derived:
            for key in self.derived_quantity_keys:
                arr = self.get_data(key)
                out[key] = arr.copy() if copy else arr
        return out

    def to_dataframe(self) -> pd.DataFrame:
        d = self.to_dict()
        d.pop("z")
        return pd.DataFrame.from_dict(d)
