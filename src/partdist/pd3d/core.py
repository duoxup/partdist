from __future__ import annotations

from typing import Dict, Mapping, Sequence, Union

import numpy as np
from xtils import g_c, g_m0, g_e0, relconv

from ..particle_array_quantity import (
    ParticleArrayQuantity,
    QuantityCategory,
    QuantityDTypeKind,
)

from .analysis import compute_twiss_plane


ArrayLike = Union[float, Sequence[float], np.ndarray]


def _as_1d_array(a: ArrayLike, name: str) -> np.ndarray:
    arr = np.asarray(a).reshape(-1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array, got shape {arr.shape}.")
    return arr


def _as_1d_float_array(a: ArrayLike, name: str) -> np.ndarray:
    arr = np.asarray(a, dtype=float).reshape(-1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array, got shape {arr.shape}.")
    return arr


class ParticleDistribution:
    """
    Fully object-based particle distribution container.

    Internally, all quantities are stored as ParticleArrayQuantity objects.
    Externally, convenience properties such as .x, .vx, .Q, .pz, .gamma
    are preserved and return only the underlying ndarray.

    Fixed stored quantities
    -----------------------
    x, y, z   : position [m]
    vx, vy, vz: velocity [m/s]
    t         : time [s]
    Q         : macro-particle charge [C]

    Optional extras
    ---------------
    Any other particle-resolved quantities are stored as extra quantities.
    """

    _BASE_SPECS: dict[str, dict] = {
        "x":  {"unit": "m",   "dtype_kind": "float", "short_name": "x",   "long_name": "horizontal position",   "latex_name": r"$x$",   "category": "position", "is_derived": False},
        "y":  {"unit": "m",   "dtype_kind": "float", "short_name": "y",   "long_name": "vertical position",     "latex_name": r"$y$",   "category": "position", "is_derived": False},
        "z":  {"unit": "m",   "dtype_kind": "float", "short_name": "z",   "long_name": "longitudinal position", "latex_name": r"$z$",   "category": "position", "is_derived": False},
        "vx": {"unit": "m/s", "dtype_kind": "float", "short_name": "vx",  "long_name": "horizontal velocity",   "latex_name": r"$v_x$", "category": "velocity", "is_derived": False},
        "vy": {"unit": "m/s", "dtype_kind": "float", "short_name": "vy",  "long_name": "vertical velocity",     "latex_name": r"$v_y$", "category": "velocity", "is_derived": False},
        "vz": {"unit": "m/s", "dtype_kind": "float", "short_name": "vz",  "long_name": "longitudinal velocity", "latex_name": r"$v_z$", "category": "velocity", "is_derived": False},
        "t":  {"unit": "s",   "dtype_kind": "float", "short_name": "t",   "long_name": "time",                  "latex_name": r"$t$",   "category": "time",     "is_derived": False},
        "Q":  {"unit": "C",   "dtype_kind": "float", "short_name": "Q",   "long_name": "macro-particle charge", "latex_name": r"$Q$",   "category": "charge",   "is_derived": False},
    }

    _DERIVED_SPECS: dict[str, dict] = {
        "Q_abs":              {"unit": "C",      "dtype_kind": "float", "short_name": "abs. Q",    "long_name": "macro-particle charge",         "latex_name": r"$|Q|$",       "category": "charge",   "is_derived": True},
        "radial_position":    {"unit": "m",      "dtype_kind": "float", "short_name": "r",         "long_name": "radial position",               "latex_name": r"$r$",         "category": "geometry", "is_derived": True},
        "transverse_speed":   {"unit": "m/s",    "dtype_kind": "float", "short_name": "v_perp",    "long_name": "transverse speed",              "latex_name": r"$v_{\perp}$", "category": "geometry", "is_derived": True},
        "radial_velocity":    {"unit": "m/s",    "dtype_kind": "float", "short_name": "v_r",       "long_name": "radial velocity",               "latex_name": r"$v_r$",       "category": "geometry", "is_derived": True},
        "azimuthal_velocity": {"unit": "m/s",    "dtype_kind": "float", "short_name": "v_phi",     "long_name": "azimuthal velocity",            "latex_name": r"$v_{\phi}$",  "category": "geometry", "is_derived": True},
        "speed":              {"unit": "m/s",    "dtype_kind": "float", "short_name": "v",         "long_name": "speed",                         "latex_name": r"$v$",         "category": "velocity", "is_derived": True},
        "beta":               {"unit": "",       "dtype_kind": "float", "short_name": "beta",      "long_name": "normalized speed",              "latex_name": r"$\beta$",     "category": "velocity", "is_derived": True},
        "gamma":              {"unit": "",       "dtype_kind": "float", "short_name": "gamma",     "long_name": "Lorentz factor",                "latex_name": r"$\gamma$",    "category": "velocity", "is_derived": True},
        "px":                 {"unit": "eV/c",   "dtype_kind": "float", "short_name": "px",        "long_name": "horizontal momentum",           "latex_name": r"$p_x$",       "category": "momentum", "is_derived": True},
        "py":                 {"unit": "eV/c",   "dtype_kind": "float", "short_name": "py",        "long_name": "vertical momentum",             "latex_name": r"$p_y$",       "category": "momentum", "is_derived": True},
        "pz":                 {"unit": "eV/c",   "dtype_kind": "float", "short_name": "pz",        "long_name": "longitudinal momentum",         "latex_name": r"$p_z$",       "category": "momentum", "is_derived": True},
        "xp":                 {"unit": "rad",    "dtype_kind": "float", "short_name": "xp",        "long_name": "horizontal normlized angle",    "latex_name": r"$x'$",        "category": "other",    "is_derived": True},
        "yp":                 {"unit": "rad",    "dtype_kind": "float", "short_name": "yp",        "long_name": "vertical normlized angle",      "latex_name": r"$y'$",        "category": "other",    "is_derived": True},
        "p_abs_si":           {"unit": "kg*m/s", "dtype_kind": "float", "short_name": "p",         "long_name": "momentum magnitude (SI)",       "latex_name": r"$|p|$",       "category": "momentum", "is_derived": True},
        "p_abs":              {"unit": "eV/c",   "dtype_kind": "float", "short_name": "p",         "long_name": "momentum magnitude",            "latex_name": r"$|p|$",       "category": "momentum", "is_derived": True},
        "kinetic_energy":     {"unit": "J",      "dtype_kind": "float", "short_name": "Ek",        "long_name": "kinetic energy",                "latex_name": r"$E_k$",       "category": "energy",   "is_derived": True},
        "kinetic_energy_eV":  {"unit": "eV",     "dtype_kind": "float", "short_name": "Ek",        "long_name": "kinetic energy",                "latex_name": r"$E_k$",       "category": "energy",   "is_derived": True},
        "current_x":          {"unit": "C*m/s",  "dtype_kind": "float", "short_name": "I_x_like",  "long_name": "x current-like weight",         "latex_name": r"$Qv_x$",      "category": "current",  "is_derived": True},
        "current_y":          {"unit": "C*m/s",  "dtype_kind": "float", "short_name": "I_y_like",  "long_name": "y current-like weight",         "latex_name": r"$Qv_y$",      "category": "current",  "is_derived": True},
        "current_z":          {"unit": "C*m/s",  "dtype_kind": "float", "short_name": "I_z_like",  "long_name": "z current-like weight",         "latex_name": r"$Qv_z$",      "category": "current",  "is_derived": True},
        "current_abs":        {"unit": "C*m/s",  "dtype_kind": "float", "short_name": "I_like",    "long_name": "current-like weight magnitude", "latex_name": r"$|Qv|$",      "category": "current",  "is_derived": True},
        "current_x_abs":      {"unit": "C*m/s",  "dtype_kind": "float", "short_name": "|I_x_like|","long_name": "absolute x current-like weight", "latex_name": r"$|Qv_x|$",    "category": "current",  "is_derived": True},
        "current_y_abs":      {"unit": "C*m/s",  "dtype_kind": "float", "short_name": "|I_y_like|","long_name": "absolute y current-like weight", "latex_name": r"$|Qv_y|$",    "category": "current",  "is_derived": True},
        "current_z_abs":      {"unit": "C*m/s",  "dtype_kind": "float", "short_name": "|I_z_like|","long_name": "absolute z current-like weight", "latex_name": r"$|Qv_z|$",    "category": "current",  "is_derived": True},
    }

    def __init__(
        self,
        *,
        x: ArrayLike,
        y: ArrayLike,
        z: ArrayLike,
        vx: ArrayLike,
        vy: ArrayLike,
        vz: ArrayLike,
        t: ArrayLike,
        Q: ArrayLike,
        extras: Mapping[str, ArrayLike] | Mapping[str, ParticleArrayQuantity] | None = None,
    ) -> None:
        self._quantities: Dict[str, ParticleArrayQuantity] = {}

        base_arrays = {
            "x": _as_1d_float_array(x, "x"),
            "y": _as_1d_float_array(y, "y"),
            "z": _as_1d_float_array(z, "z"),
            "vx": _as_1d_float_array(vx, "vx"),
            "vy": _as_1d_float_array(vy, "vy"),
            "vz": _as_1d_float_array(vz, "vz"),
            "t": _as_1d_float_array(t, "t"),
            "Q": _as_1d_float_array(Q, "Q"),
        }

        n = len(base_arrays["x"])
        for key, arr in base_arrays.items():
            if len(arr) != n:
                raise ValueError(f"All base arrays must have the same length; {key} mismatches.")

        for key, arr in base_arrays.items():
            spec = self._BASE_SPECS[key]
            self._quantities[key] = ParticleArrayQuantity(
                name=key,
                data=arr,
                unit=spec["unit"],
                dtype_kind=spec["dtype_kind"],
                short_name=spec["short_name"],
                long_name=spec["long_name"],
                latex_name=spec["latex_name"],
                category=spec["category"],
                is_derived=spec["is_derived"],
            )

        if extras is not None:
            for key, value in extras.items():
                self.add_quantity(key, value, inplace=True)
                
    def __len__(self) -> int:
        return self.n

    @classmethod
    def from_arrays(
        cls,
        x: ArrayLike,
        y: ArrayLike,
        z: ArrayLike,
        vx: ArrayLike,
        vy: ArrayLike,
        vz: ArrayLike,
        t: ArrayLike,
        Q: ArrayLike | None = None,
        extras: Mapping[str, ArrayLike] | Mapping[str, ParticleArrayQuantity] | None = None,
    ) -> "ParticleDistribution":
        x = _as_1d_float_array(x, "x")
        n = len(x)
        if Q is None:
            Q = np.ones(n, dtype=float)
        return cls(x=x, y=y, z=z, vx=vx, vy=vy, vz=vz, t=t, Q=Q, extras=extras)

    @classmethod
    def from_dict(cls, data: Mapping[str, ArrayLike]) -> "ParticleDistribution":
        required = ("x", "y", "z", "vx", "vy", "vz", "t", "Q")
        missing = [k for k in required if k not in data]
        if missing:
            raise KeyError(f"Missing required keys: {missing}")
        extras = {k: v for k, v in data.items() if k not in required}
        return cls(
            x=data["x"],
            y=data["y"],
            z=data["z"],
            vx=data["vx"],
            vy=data["vy"],
            vz=data["vz"],
            t=data["t"],
            Q=data["Q"],
            extras=extras,
        )

    @property
    def base_quantity_keys(self) -> tuple[str, ...]:
        return tuple(self._BASE_SPECS.keys())

    @property
    def derived_quantity_keys(self) -> tuple[str, ...]:
        return tuple(self._DERIVED_SPECS.keys())

    @property
    def extra_quantity_keys(self) -> tuple[str, ...]:
        return tuple(
            k for k in self._quantities
            if k not in self._BASE_SPECS and k not in self._DERIVED_SPECS
        )

    @property
    def quantity_keys(self) -> tuple[str, ...]:
        return self.base_quantity_keys + self.derived_quantity_keys + self.extra_quantity_keys

    @property
    def particle_quantity_keys(self) -> tuple[str, ...]:
        return self.quantity_keys

    def has_quantity(self, key: str) -> bool:
        return key in self.quantity_keys

    def quantity_kind(self, key: str) -> str:
        if key in self._BASE_SPECS:
            return "base"
        if key in self._DERIVED_SPECS:
            return "derived"
        if key in self._quantities:
            return "extra"
        raise KeyError(f"Unknown quantity: {key!r}")

    @property
    def n(self) -> int:
        return self._quantities["x"].n

    @property
    def size(self) -> int:
        return self.n

    @property
    def total_charge(self) -> float:
        return float(np.sum(self.Q))

    @property
    def total_charge_abs(self) -> float:
        return float(np.sum(np.abs(self.Q)))

    def _make_derived_quantity(self, key: str) -> ParticleArrayQuantity:
        if key not in self._DERIVED_SPECS:
            raise KeyError(f"Unknown derived quantity: {key!r}")

        spec = self._DERIVED_SPECS[key]
        data = getattr(self, f"_calc_{key}")()

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

    def get_quantity(self, key: str) -> ParticleArrayQuantity:
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
    ) -> "ParticleDistribution":
        """
        Add a new stored quantity.
    
        Notes
        -----
        - Base quantity names are forbidden.
        - Derived quantity names are forbidden.
        - Existing quantity names are forbidden.
        """
        if key in self._BASE_SPECS or key in self._DERIVED_SPECS:
            raise ValueError(f"'{key}' conflicts with a built-in quantity name.")
    
        if key in self._quantities:
            raise ValueError(f"Quantity '{key}' already exists. Use update_quantity(...) instead.")
    
        out = self if inplace else self.copy()
    
        if isinstance(value, ParticleArrayQuantity):
            q = value.copy()
            if q.name != key:
                q.name = key
        else:
            arr = _as_1d_array(value, key)
            q = ParticleArrayQuantity(
                name=key,
                data=arr,
                unit=unit,
                dtype_kind=dtype_kind,
                short_name=short_name,
                long_name=long_name,
                latex_name=latex_name,
                category=category,
                is_derived=is_derived,
                is_discrete=is_discrete,
                preferred_scale=preferred_scale,
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
    ) -> "ParticleDistribution":
        """
        Update an existing stored quantity.
    
        Supported
        ---------
        - base quantities
        - extra quantities
    
        Not supported
        -------------
        - derived quantities
    
        Parameters
        ----------
        key
            Quantity name to update.
        value
            New data, or a ParticleArrayQuantity object.
        update_meta
            Only relevant for extra quantities when `value` is a ParticleArrayQuantity.
    
            - False: keep existing metadata, replace only data
            - True : replace the whole quantity object including metadata
    
            For base quantities, metadata is always preserved.
        inplace
            If True, modify this object in place.
        """
        if key in self._DERIVED_SPECS:
            raise ValueError(
                f"Cannot update derived quantity '{key}' directly. "
                "Please update the underlying base quantities instead."
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
            arr = _as_1d_array(value, key)
            new_q = ParticleArrayQuantity(
                name=key,
                data=arr,
                unit=old_q.unit,
                dtype_kind=old_q.dtype_kind,
                short_name=old_q.short_name,
                long_name=old_q.long_name,
                latex_name=old_q.latex_name,
                category=old_q.category,
                is_derived=old_q.is_derived,
                is_discrete=old_q.is_discrete,
                preferred_scale=old_q.preferred_scale,
            )
    
        if new_q.n != out.n:
            raise ValueError(f"Quantity '{key}' must have length {out.n}, got {new_q.n}.")
    
        if key in self._BASE_SPECS:
            # Base quantities: only data may be updated, metadata is fixed.
            out._quantities[key].data = np.asarray(new_q.data, dtype=float).reshape(-1)
            return out
    
        # Extra quantities
        if update_meta and isinstance(value, ParticleArrayQuantity):
            out._quantities[key] = new_q
        else:
            out._quantities[key].data = new_q.data
    
        return out
    
    
    def drop_quantity(self, key: str, *, inplace: bool = True) -> "ParticleDistribution":
        """
        Drop an existing extra quantity.
    
        Notes
        -----
        - Base quantities cannot be dropped.
        - Derived quantities are not stored and therefore cannot be dropped.
        """
        if key in self._BASE_SPECS:
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
    ) -> "ParticleDistribution":
        """
        Add a new extra quantity.
    
        This is a thin wrapper over add_quantity(...).
        """
        if key in self._BASE_SPECS or key in self._DERIVED_SPECS:
            raise ValueError(f"'{key}' conflicts with a built-in quantity name.")
        if key in self._quantities:
            raise ValueError(f"Extra quantity '{key}' already exists. Use update_extra(...) instead.")
    
        return self.add_quantity(
            key,
            value,
            unit=unit,
            dtype_kind=dtype_kind,
            short_name=short_name,
            long_name=long_name,
            latex_name=latex_name,
            category=category,
            is_derived=False,
            is_discrete=is_discrete,
            preferred_scale=preferred_scale,
            inplace=inplace,
        )
    
    
    def update_extra(
        self,
        key: str,
        value: ArrayLike | ParticleArrayQuantity,
        *,
        update_meta: bool = False,
        inplace: bool = True,
    ) -> "ParticleDistribution":
        """
        Update an existing extra quantity.
        """
        if key in self._BASE_SPECS:
            raise ValueError(f"'{key}' is a base quantity, not an extra quantity.")
        if key in self._DERIVED_SPECS:
            raise ValueError(f"'{key}' is a derived quantity, not an extra quantity.")
        if key not in self._quantities:
            raise KeyError(f"Extra quantity '{key}' does not exist. Use add_extra(...) instead.")
    
        return self.update_quantity(
            key,
            value,
            update_meta=update_meta,
            inplace=inplace,
        )
    
    
    def drop_extra(self, key: str, *, inplace: bool = True) -> "ParticleDistribution":
        """
        Drop an existing extra quantity.
        """
        if key in self._BASE_SPECS:
            raise ValueError(f"'{key}' is a base quantity, not an extra quantity.")
        if key in self._DERIVED_SPECS:
            raise ValueError(f"'{key}' is a derived quantity, not an extra quantity.")
        if key not in self._quantities:
            raise KeyError(f"Extra quantity '{key}' not found.")
    
        return self.drop_quantity(key, inplace=inplace)
    

    def copy(self) -> "ParticleDistribution":
        extras = {k: q.copy() for k, q in self._quantities.items() if k not in self._BASE_SPECS}
        return ParticleDistribution(
            x=self.x.copy(),
            y=self.y.copy(),
            z=self.z.copy(),
            vx=self.vx.copy(),
            vy=self.vy.copy(),
            vz=self.vz.copy(),
            t=self.t.copy(),
            Q=self.Q.copy(),
            extras=extras,
        )
    
    def update_data(
        self,
        key: str,
        data: ArrayLike,
        *,
        inplace: bool = False,
    ) -> "ParticleDistribution":
        """
        Update only the data array of an existing quantity.
        """
        return self.update_quantity(key, data, update_meta=False, inplace=inplace)

    def to_dict(
        self,
        *,
        include_extras: bool = True,
        include_derived: bool = False,
        copy: bool = True,
    ) -> Dict[str, np.ndarray]:
        out = {}
        for key in self.base_quantity_keys:
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

    def slice(self, mask: np.ndarray | slice | ArrayLike) -> "ParticleDistribution":
        extras = {}
        for key in self.extra_quantity_keys:
            q = self.get_quantity(key)
            extras[key] = ParticleArrayQuantity(
                name=q.name,
                data=q.data[mask],
                unit=q.unit,
                dtype_kind=q.dtype_kind,
                short_name=q.short_name,
                long_name=q.long_name,
                latex_name=q.latex_name,
                category=q.category,
                is_derived=q.is_derived,
                is_discrete=q.is_discrete,
                preferred_scale=q.preferred_scale,
            )

        return ParticleDistribution(
            x=self.x[mask],
            y=self.y[mask],
            z=self.z[mask],
            vx=self.vx[mask],
            vy=self.vy[mask],
            vz=self.vz[mask],
            t=self.t[mask],
            Q=self.Q[mask],
            extras=extras,
        )

    def sort_by(self, key: str, *, ascending: bool = True) -> "ParticleDistribution":
        values = self.get_data(key)
        order = np.argsort(values)
        if not ascending:
            order = order[::-1]
        return self.slice(order)

    def _get_weights(self, weight: str | np.ndarray | None = "absQ") -> np.ndarray:
        if weight is None:
            w = np.ones(self.size, dtype=float)
        elif isinstance(weight, str):
            if weight == "Q":
                w = np.asarray(self.Q, dtype=float)
            elif weight == "absQ":
                w = np.abs(np.asarray(self.Q, dtype=float))
            else:
                w = np.asarray(self.get_data(weight), dtype=float)
        else:
            w = np.asarray(weight, dtype=float)

        if w.ndim != 1 or len(w) != self.size:
            raise ValueError(f"Weights must be a 1D array of length {self.size}.")
        return w

    def mean(self, key: str, weight: str | np.ndarray | None = "absQ") -> float:
        x = np.asarray(self.get_data(key), dtype=float)
        w = self._get_weights(weight)
        wsum = np.sum(w)
        if wsum == 0.0:
            raise ValueError("Sum of weights is zero.")
        return float(np.sum(w * x) / wsum)

    def var(self, key: str, weight: str | np.ndarray | None = "absQ") -> float:
        x = np.asarray(self.get_data(key), dtype=float)
        w = self._get_weights(weight)
        mu = self.mean(key, weight=weight)
        wsum = np.sum(w)
        if wsum == 0.0:
            raise ValueError("Sum of weights is zero.")
        return float(np.sum(w * (x - mu) ** 2) / wsum)

    def std(self, key: str, weight: str | np.ndarray | None = "absQ") -> float:
        return float(np.sqrt(self.var(key, weight=weight)))

    def rms(self, key: str, weight: str | np.ndarray | None = "absQ") -> float:
        x = np.asarray(self.get_data(key), dtype=float)
        w = self._get_weights(weight)
        wsum = np.sum(w)
        if wsum == 0.0:
            raise ValueError("Sum of weights is zero.")
        return float(np.sqrt(np.sum(w * x**2) / wsum))

    def covariance(self, key1: str, key2: str, weight: str | np.ndarray | None = "absQ") -> float:
        x1 = np.asarray(self.get_data(key1), dtype=float)
        x2 = np.asarray(self.get_data(key2), dtype=float)
        w = self._get_weights(weight)
        mu1 = self.mean(key1, weight=weight)
        mu2 = self.mean(key2, weight=weight)
        wsum = np.sum(w)
        if wsum == 0.0:
            raise ValueError("Sum of weights is zero.")
        return float(np.sum(w * (x1 - mu1) * (x2 - mu2)) / wsum)

    def correlation(self, key1: str, key2: str, weight: str | np.ndarray | None = "absQ") -> float:
        cov = self.covariance(key1, key2, weight=weight)
        s1 = self.std(key1, weight=weight)
        s2 = self.std(key2, weight=weight)
        if s1 == 0.0 or s2 == 0.0:
            return float("nan")
        return float(cov / (s1 * s2))

    def linear_fit(self, xkey: str, ykey: str, weight: str | np.ndarray | None = "absQ") -> tuple[float, float]:
        x = np.asarray(self.get_data(xkey), dtype=float)
        y = np.asarray(self.get_data(ykey), dtype=float)
        w = self._get_weights(weight)

        x_mean = self.mean(xkey, weight=weight)
        y_mean = self.mean(ykey, weight=weight)

        dx = x - x_mean
        var_x = np.sum(w * dx**2)
        if var_x == 0.0:
            raise ValueError(f"Cannot fit {ykey} vs {xkey}: zero variance in {xkey}.")

        cov_xy = np.sum(w * dx * (y - y_mean))
        slope = cov_xy / var_x
        intercept = y_mean - slope * x_mean
        return float(slope), float(intercept)

    def centroid(self, weight: str | np.ndarray | None = "absQ") -> dict[str, float]:
        return {
            "x": self.mean("x", weight=weight),
            "y": self.mean("y", weight=weight),
            "z": self.mean("z", weight=weight),
            "vx": self.mean("vx", weight=weight),
            "vy": self.mean("vy", weight=weight),
            "vz": self.mean("vz", weight=weight),
            "t": self.mean("t", weight=weight),
            "Q": self.mean("Q", weight=weight),
        }

    def sigma_dict(self, weight: str | np.ndarray | None = "absQ") -> dict[str, float]:
        return {
            "x": self.std("x", weight=weight),
            "y": self.std("y", weight=weight),
            "z": self.std("z", weight=weight),
            "vx": self.std("vx", weight=weight),
            "vy": self.std("vy", weight=weight),
            "vz": self.std("vz", weight=weight),
            "t": self.std("t", weight=weight),
            "Q": self.std("Q", weight=weight),
        }

    def _calc_radial_position(self) -> np.ndarray:
        return np.sqrt(self.x**2 + self.y**2)

    def _calc_transverse_speed(self) -> np.ndarray:
        return np.sqrt(self.vx**2 + self.vy**2)

    def _calc_radial_velocity(self) -> np.ndarray:
        r = self.radial_position + 1e-30
        return (self.x * self.vx + self.y * self.vy) / r

    def _calc_azimuthal_velocity(self) -> np.ndarray:
        r = self.radial_position + 1e-30
        return (self.x * self.vy - self.y * self.vx) / r

    def _calc_speed(self) -> np.ndarray:
        return np.sqrt(self.vx**2 + self.vy**2 + self.vz**2)

    def _calc_beta(self) -> np.ndarray:
        return np.minimum(self.speed / g_c, 1.0 - 1e-15)

    def _calc_gamma(self) -> np.ndarray:
        b = self.beta
        return 1.0 / np.sqrt(1.0 - b**2)

    def momentum_si(self, *, m0: float = g_m0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        g = self.gamma
        return g * m0 * self.vx, g * m0 * self.vy, g * m0 * self.vz

    def _momentum_evc_backup(self, *, m0: float = g_m0, e0: float = g_e0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        px_si, py_si, pz_si = self.momentum_si(m0=m0)
        factor = g_c / abs(e0)
        return px_si * factor, py_si * factor, pz_si * factor
    
    def momentum_evc(self, *, m0: float = g_m0, e0: float = g_e0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        from .utils import _velocity_components_to_momentum_evc
        px_evc, py_evc, pz_evc = _velocity_components_to_momentum_evc(
            self.vx, self.vy, self.vz,
            m0=m0, q=e0)
        return px_evc, py_evc, pz_evc
    
    def _calc_Q_abs(self) -> np.ndarray:
        return np.abs(self.Q)
    
    def _calc_px(self) -> np.ndarray:
        return self.momentum_evc()[0]

    def _calc_py(self) -> np.ndarray:
        return self.momentum_evc()[1]

    def _calc_pz(self) -> np.ndarray:
        return self.momentum_evc()[2]
    
    def _calc_xp(self) -> np.ndarray:
        return self.px/self.p_abs
    
    def _calc_yp(self) -> np.ndarray:
        return self.py/self.p_abs

    def _calc_p_abs_si(self) -> np.ndarray:
        px, py, pz = self.momentum_si()
        return np.sqrt(px**2 + py**2 + pz**2)

    def _calc_p_abs(self) -> np.ndarray:
        px, py, pz = self.momentum_evc()
        return np.sqrt(px**2 + py**2 + pz**2)

    def _calc_kinetic_energy(self) -> np.ndarray:
        return (self.gamma - 1.0) * (g_m0 * g_c**2)

    def _calc_kinetic_energy_eV(self) -> np.ndarray:
        return self.kinetic_energy / abs(g_e0)

    def _calc_current_x(self) -> np.ndarray:
        return self.Q * self.vx

    def _calc_current_y(self) -> np.ndarray:
        return self.Q * self.vy

    def _calc_current_z(self) -> np.ndarray:
        return self.Q * self.vz

    def _calc_current_abs(self) -> np.ndarray:
        return np.sqrt(self.current_x**2 + self.current_y**2 + self.current_z**2)

    def _calc_current_x_abs(self) -> np.ndarray:
        return np.abs(self.current_x)

    def _calc_current_y_abs(self) -> np.ndarray:
        return np.abs(self.current_y)

    def _calc_current_z_abs(self) -> np.ndarray:
        return np.abs(self.current_z)

    #%% shortcuts for statistics
    @property
    def gamma0(self) -> float:
        return float(relconv.gamma_from_ke_eV(self.mean('kinetic_energy_eV')))
    
    @property
    def beta0(self) -> float:
        return float(relconv.beta_from_ke_eV(self.mean('kinetic_energy_eV')))
    
    @property
    def chirp(self) -> float:  #eV/m
        return float(self.covariance('z', 'kinetic_energy_eV')/self.std('z')**2)
    
    @property
    def cor_ekin(self) -> float: #eV
        return float(self.covariance('z', 'kinetic_energy_eV')/self.std('z'))
    
    @property
    def emit_x(self) -> float:
        var_x = self.var('x')
        var_xp = self.var('xp')
        cov_xxp = self.covariance('x', 'xp')
        return float(np.sqrt(var_x * var_xp - cov_xxp**2))
    
    @property
    def emit_y(self) -> float:
        var_y = self.var('y')
        var_yp = self.var('yp')
        cov_yyp = self.covariance('y', 'yp')
        return float(np.sqrt(var_y * var_yp - cov_yyp**2))
    
    @property
    def Nemit_x(self) -> float:
        return float(self.beta0*self.gamma0*self.emit_x)
    
    @property
    def Nemit_y(self) -> float:
        return float(self.beta0*self.gamma0*self.emit_y)
    
    @property
    def alpha_x(self) -> float:
        # return float(-self.covariance('x', 'xp') / self.emit_x)
        return compute_twiss_plane(self, plane='x', weight='Q_abs').alpha
    
    @property
    def beta_x(self) -> float:
        # return float(self.var('x') / self.emit_x)
        return compute_twiss_plane(self, plane='x', weight='Q_abs').beta
    
    @property
    def gamma_x(self) -> float:
        # return float(self.var('xp') / self.emit_x)
        return compute_twiss_plane(self, plane='x', weight='Q_abs').gamma_twiss
    
    @property
    def alpha_y(self) -> float:
        # return float(-self.covariance('y', 'yp') / self.emit_y)
        return compute_twiss_plane(self, plane='y', weight='Q_abs').alpha
    
    @property
    def beta_y(self) -> float:
        # return float(self.var('y') / self.emit_y)
        return compute_twiss_plane(self, plane='y', weight='Q_abs').beta
    
    @property
    def gamma_y(self) -> float:
        # return float(self.var('yp') / self.emit_y)
        return compute_twiss_plane(self, plane='y', weight='Q_abs').gamma_twiss
    
    @property
    def twiss(self) -> Dict[str, float]:
        return dict(alpha_x = self.alpha_x,
                    beta_x = self.beta_x,
                    alpha_y = self.alpha_y,
                    beta_y = self.beta_y)
    
    @property 
    def cor_pz(self) -> float: #eV/c non-standard name
        return float(self.covariance('z', 'pz', 'absQ')/self.std('z'))

    #%% legacy properties
    @property
    def x(self) -> np.ndarray:
        return self._quantities["x"].data

    @property
    def y(self) -> np.ndarray:
        return self._quantities["y"].data

    @property
    def z(self) -> np.ndarray:
        return self._quantities["z"].data

    @property
    def vx(self) -> np.ndarray:
        return self._quantities["vx"].data

    @property
    def vy(self) -> np.ndarray:
        return self._quantities["vy"].data

    @property
    def vz(self) -> np.ndarray:
        return self._quantities["vz"].data

    @property
    def t(self) -> np.ndarray:
        return self._quantities["t"].data

    @property
    def Q(self) -> np.ndarray:
        return self._quantities["Q"].data
    
    @property
    def Q_abs(self) -> np.ndarray:
        return self._calc_Q_abs()

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
    def px(self) -> np.ndarray:
        return self._calc_px()

    @property
    def py(self) -> np.ndarray:
        return self._calc_py()
    
    @property
    def pz(self) -> np.ndarray:
        return self._calc_pz()

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
    def current_x(self) -> np.ndarray:
        return self._calc_current_x()

    @property
    def current_y(self) -> np.ndarray:
        return self._calc_current_y()

    @property
    def current_z(self) -> np.ndarray:
        return self._calc_current_z()

    @property
    def current_abs(self) -> np.ndarray:
        return self._calc_current_abs()

    @property
    def current_x_abs(self) -> np.ndarray:
        return self._calc_current_x_abs()

    @property
    def current_y_abs(self) -> np.ndarray:
        return self._calc_current_y_abs()

    @property
    def current_z_abs(self) -> np.ndarray:
        return self._calc_current_z_abs()

    @property
    def extras(self) -> Dict[str, ParticleArrayQuantity]:
        return {k: self._quantities[k] for k in self.extra_quantity_keys}

    @property
    def id(self) -> np.ndarray:
        return self.get_data("pid")

    @property
    def pid(self) -> np.ndarray:
        return self.id