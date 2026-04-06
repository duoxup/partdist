# Momentum as Base Storage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `vx/vy/vz` (m/s) with `px/py/pz` (eV/c) as the canonical stored quantities in `ParticleDistribution`, making `vx/vy/vz` derived, and rebuilding the γ/β computation chain from momentum for better numerical stability at high energies.

**Architecture:** Four files change in dependency order: `core.py` (data model) → `utils.py` (helpers) → `io.py` (file I/O) → `manipulator.py` (operations). `analysis.py` and `viz.py` access data through `dist.get_data(key)` and derived properties and require no changes.

**Tech Stack:** Python 3.10+, numpy, xtils (relconv, g_m0, g_e0, g_c)

---

## File Map

| File | Change |
|------|--------|
| `src/partdist/pd3d/core.py` | Swap base/derived specs, update `__init__`/`from_arrays`/`from_dict`/`copy`/`slice`/`centroid`/`sigma_dict`, rewrite computation chain, update property accessors |
| `src/partdist/pd3d/utils.py` | Rename `_update_velocity_components` → `_update_momentum_components`, simplify `_replace_velocity_from_momentum` |
| `src/partdist/pd3d/io.py` | Remove redundant velocity↔momentum conversions on read/write paths, update internal `ParticleDistribution` construction |
| `src/partdist/pd3d/manipulator.py` | Update `match_twiss_plane` to use `px/pz` slope, simplify `_update_quantity_array`, update `replicate_longitudinally` base keys |
| `tests/test_core3d.py` | Update construction from `vx/vy/vz` to `px/py/pz` |

---

## Task 1: core.py — Base/Derived Swap and New Computation Chain

**Files:**
- Modify: `src/partdist/pd3d/core.py`
- Modify: `tests/test_core3d.py`

- [ ] **Step 1: Update `tests/test_core3d.py` to use the new `px/py/pz` interface (will fail until Step 2)**

Replace the module-level setup and dist construction:

```python
#!/usr/bin/env python3
import numpy as np
from partdist.pd3d.core import ParticleDistribution
from partdist.pd3d.io import velocity_to_momentum_evc

x = np.array([0.0, 1e-6, -1e-6])
y = np.array([0.0, 2e-6, -2e-6])
z = np.array([0.0, 1e-3, -1e-3])

vx = np.array([0.0, 1e5, -1e5])
vy = np.array([0.0, 2e5, -2e5])
vz = np.array([2.9e8, 2.9e8, 2.9e8])
px, py, pz = velocity_to_momentum_evc(vx, vy, vz)

t = np.array([0.0, 3e-12, -3e-12])
Q = np.array([-1e-12, -1e-12, -1e-12])
pid = np.array([10, 11, 12], dtype=np.int64)

dist = ParticleDistribution.from_arrays(
    x=x, y=y, z=z,
    px=px, py=py, pz=pz,
    t=t, Q=Q,
    extras={"pid": pid},
)

print(dist.particle_quantity_keys)
print(dist.mean("z"))
print(dist.std("z"))
print(dist.pz)
print(dist.kinetic_energy_eV)
print(dist.id)

paq = dist.get_quantity('x')
```

- [ ] **Step 2: Update `_BASE_SPECS` in `core.py`**

Replace the `vx/vy/vz` entries with `px/py/pz`:

```python
_BASE_SPECS: dict[str, dict] = {
    "x":  {"unit": "m",     "dtype_kind": "float", "short_name": "x",  "long_name": "horizontal position",   "latex_name": r"$x$",   "category": "position", "is_derived": False},
    "y":  {"unit": "m",     "dtype_kind": "float", "short_name": "y",  "long_name": "vertical position",     "latex_name": r"$y$",   "category": "position", "is_derived": False},
    "z":  {"unit": "m",     "dtype_kind": "float", "short_name": "z",  "long_name": "longitudinal position", "latex_name": r"$z$",   "category": "position", "is_derived": False},
    "px": {"unit": "eV/c",  "dtype_kind": "float", "short_name": "px", "long_name": "horizontal momentum",   "latex_name": r"$p_x$", "category": "momentum", "is_derived": False},
    "py": {"unit": "eV/c",  "dtype_kind": "float", "short_name": "py", "long_name": "vertical momentum",     "latex_name": r"$p_y$", "category": "momentum", "is_derived": False},
    "pz": {"unit": "eV/c",  "dtype_kind": "float", "short_name": "pz", "long_name": "longitudinal momentum", "latex_name": r"$p_z$", "category": "momentum", "is_derived": False},
    "t":  {"unit": "s",     "dtype_kind": "float", "short_name": "t",  "long_name": "time",                  "latex_name": r"$t$",   "category": "time",     "is_derived": False},
    "Q":  {"unit": "C",     "dtype_kind": "float", "short_name": "Q",  "long_name": "macro-particle charge", "latex_name": r"$Q$",   "category": "charge",   "is_derived": False},
}
```

- [ ] **Step 3: Update `_DERIVED_SPECS` in `core.py`**

Remove `px/py/pz` entries; add `vx/vy/vz` entries. The existing `_DERIVED_SPECS` block becomes:

```python
_DERIVED_SPECS: dict[str, dict] = {
    "Q_abs":              {"unit": "C",      "dtype_kind": "float", "short_name": "abs. Q",    "long_name": "macro-particle charge",         "latex_name": r"$|Q|$",       "category": "charge",   "is_derived": True},
    "vx":                 {"unit": "m/s",    "dtype_kind": "float", "short_name": "vx",        "long_name": "horizontal velocity",           "latex_name": r"$v_x$",       "category": "velocity", "is_derived": True},
    "vy":                 {"unit": "m/s",    "dtype_kind": "float", "short_name": "vy",        "long_name": "vertical velocity",             "latex_name": r"$v_y$",       "category": "velocity", "is_derived": True},
    "vz":                 {"unit": "m/s",    "dtype_kind": "float", "short_name": "vz",        "long_name": "longitudinal velocity",         "latex_name": r"$v_z$",       "category": "velocity", "is_derived": True},
    "radial_position":    {"unit": "m",      "dtype_kind": "float", "short_name": "r",         "long_name": "radial position",               "latex_name": r"$r$",         "category": "geometry", "is_derived": True},
    "transverse_speed":   {"unit": "m/s",    "dtype_kind": "float", "short_name": "v_perp",    "long_name": "transverse speed",              "latex_name": r"$v_{\perp}$", "category": "geometry", "is_derived": True},
    "radial_velocity":    {"unit": "m/s",    "dtype_kind": "float", "short_name": "v_r",       "long_name": "radial velocity",               "latex_name": r"$v_r$",       "category": "geometry", "is_derived": True},
    "azimuthal_velocity": {"unit": "m/s",    "dtype_kind": "float", "short_name": "v_phi",     "long_name": "azimuthal velocity",            "latex_name": r"$v_{\phi}$",  "category": "geometry", "is_derived": True},
    "speed":              {"unit": "m/s",    "dtype_kind": "float", "short_name": "v",         "long_name": "speed",                         "latex_name": r"$v$",         "category": "velocity", "is_derived": True},
    "beta":               {"unit": "",       "dtype_kind": "float", "short_name": "beta",      "long_name": "normalized speed",              "latex_name": r"$\beta$",     "category": "velocity", "is_derived": True},
    "gamma":              {"unit": "",       "dtype_kind": "float", "short_name": "gamma",     "long_name": "Lorentz factor",                "latex_name": r"$\gamma$",    "category": "velocity", "is_derived": True},
    "xp":                 {"unit": "rad",    "dtype_kind": "float", "short_name": "xp",        "long_name": "horizontal normlized angle",    "latex_name": r"$x'$",        "category": "other",    "is_derived": True},
    "yp":                 {"unit": "rad",    "dtype_kind": "float", "short_name": "yp",        "long_name": "vertical normlized angle",      "latex_name": r"$y'$",        "category": "other",    "is_derived": True},
    "p_abs_si":           {"unit": "kg*m/s", "dtype_kind": "float", "short_name": "p",         "long_name": "momentum magnitude (SI)",       "latex_name": r"$|p|$",       "category": "momentum", "is_derived": True},
    "p_abs":              {"unit": "eV/c",   "dtype_kind": "float", "short_name": "p",         "long_name": "momentum magnitude",            "latex_name": r"$|p|$",       "category": "momentum", "is_derived": True},
    "kinetic_energy":     {"unit": "J",      "dtype_kind": "float", "short_name": "Ek",        "long_name": "kinetic energy",                "latex_name": r"$E_k$",       "category": "energy",   "is_derived": True},
    "kinetic_energy_eV":  {"unit": "eV",     "dtype_kind": "float", "short_name": "Ek",        "long_name": "kinetic energy",                "latex_name": r"$E_k$",       "category": "energy",   "is_derived": True},
    "current_flux_x":     {"unit": "C*m/s",  "dtype_kind": "float", "short_name": "I_x_like",  "long_name": "x current-like weight",         "latex_name": r"$Qv_x$",      "category": "current",  "is_derived": True},
    "current_flux_y":     {"unit": "C*m/s",  "dtype_kind": "float", "short_name": "I_y_like",  "long_name": "y current-like weight",         "latex_name": r"$Qv_y$",      "category": "current",  "is_derived": True},
    "current_flux_z":     {"unit": "C*m/s",  "dtype_kind": "float", "short_name": "I_z_like",  "long_name": "z current-like weight",         "latex_name": r"$Qv_z$",      "category": "current",  "is_derived": True},
    "current_flux_abs":   {"unit": "C*m/s",  "dtype_kind": "float", "short_name": "I_like",    "long_name": "current-like weight magnitude", "latex_name": r"$|Qv|$",      "category": "current",  "is_derived": True},
    "current_flux_x_abs": {"unit": "C*m/s",  "dtype_kind": "float", "short_name": "|I_x_like|","long_name": "absolute x current-like weight","latex_name": r"$|Qv_x|$",    "category": "current",  "is_derived": True},
    "current_flux_y_abs": {"unit": "C*m/s",  "dtype_kind": "float", "short_name": "|I_y_like|","long_name": "absolute y current-like weight","latex_name": r"$|Qv_y|$",    "category": "current",  "is_derived": True},
    "current_flux_z_abs": {"unit": "C*m/s",  "dtype_kind": "float", "short_name": "|I_z_like|","long_name": "absolute z current-like weight","latex_name": r"$|Qv_z|$",    "category": "current",  "is_derived": True},
}
```

- [ ] **Step 4: Update `__init__` signature and body in `core.py`**

```python
def __init__(
    self,
    *,
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    px: ArrayLike,
    py: ArrayLike,
    pz: ArrayLike,
    t: ArrayLike,
    Q: ArrayLike,
    extras: Mapping[str, ArrayLike] | Mapping[str, ParticleArrayQuantity] | None = None,
) -> None:
    self._quantities: Dict[str, ParticleArrayQuantity] = {}

    base_arrays = {
        "x":  _as_1d_float_array(x,  "x"),
        "y":  _as_1d_float_array(y,  "y"),
        "z":  _as_1d_float_array(z,  "z"),
        "px": _as_1d_float_array(px, "px"),
        "py": _as_1d_float_array(py, "py"),
        "pz": _as_1d_float_array(pz, "pz"),
        "t":  _as_1d_float_array(t,  "t"),
        "Q":  _as_1d_float_array(Q,  "Q"),
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
```

- [ ] **Step 5: Update `from_arrays`, `from_dict`, `copy`, `slice` in `core.py`**

```python
@classmethod
def from_arrays(
    cls,
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    px: ArrayLike,
    py: ArrayLike,
    pz: ArrayLike,
    t: ArrayLike,
    Q: ArrayLike | None = None,
    extras: Mapping[str, ArrayLike] | Mapping[str, ParticleArrayQuantity] | None = None,
) -> "ParticleDistribution":
    x = _as_1d_float_array(x, "x")
    n = len(x)
    if Q is None:
        Q = np.ones(n, dtype=float)
    return cls(x=x, y=y, z=z, px=px, py=py, pz=pz, t=t, Q=Q, extras=extras)

@classmethod
def from_dict(cls, data: Mapping[str, ArrayLike]) -> "ParticleDistribution":
    required = ("x", "y", "z", "px", "py", "pz", "t", "Q")
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"Missing required keys: {missing}")
    extras = {k: v for k, v in data.items() if k not in required}
    return cls(
        x=data["x"], y=data["y"], z=data["z"],
        px=data["px"], py=data["py"], pz=data["pz"],
        t=data["t"], Q=data["Q"],
        extras=extras,
    )
```

```python
def copy(self) -> "ParticleDistribution":
    extras = {k: q.copy() for k, q in self._quantities.items() if k not in self._BASE_SPECS}
    return ParticleDistribution(
        x=self.x.copy(),
        y=self.y.copy(),
        z=self.z.copy(),
        px=self.px.copy(),
        py=self.py.copy(),
        pz=self.pz.copy(),
        t=self.t.copy(),
        Q=self.Q.copy(),
        extras=extras,
    )

def slice(self, mask: np.ndarray | slice | ArrayLike) -> "ParticleDistribution":
    extras = {}
    for key in self.extra_quantity_keys:
        q = self.get_quantity(key)
        extras[key] = ParticleArrayQuantity(
            name=q.name, data=q.data[mask], unit=q.unit,
            dtype_kind=q.dtype_kind, short_name=q.short_name,
            long_name=q.long_name, latex_name=q.latex_name,
            category=q.category, is_derived=q.is_derived,
            is_discrete=q.is_discrete, preferred_scale=q.preferred_scale,
        )
    return ParticleDistribution(
        x=self.x[mask], y=self.y[mask], z=self.z[mask],
        px=self.px[mask], py=self.py[mask], pz=self.pz[mask],
        t=self.t[mask], Q=self.Q[mask],
        extras=extras,
    )
```

- [ ] **Step 6: Update `centroid()` and `sigma_dict()` in `core.py`**

```python
def centroid(self, weight: str | np.ndarray | None = "absQ") -> dict[str, float]:
    return {
        "x":  self.mean("x",  weight=weight),
        "y":  self.mean("y",  weight=weight),
        "z":  self.mean("z",  weight=weight),
        "px": self.mean("px", weight=weight),
        "py": self.mean("py", weight=weight),
        "pz": self.mean("pz", weight=weight),
        "t":  self.mean("t",  weight=weight),
        "Q":  self.mean("Q",  weight=weight),
    }

def sigma_dict(self, weight: str | np.ndarray | None = "absQ") -> dict[str, float]:
    return {
        "x":  self.std("x",  weight=weight),
        "y":  self.std("y",  weight=weight),
        "z":  self.std("z",  weight=weight),
        "px": self.std("px", weight=weight),
        "py": self.std("py", weight=weight),
        "pz": self.std("pz", weight=weight),
        "t":  self.std("t",  weight=weight),
        "Q":  self.std("Q",  weight=weight),
    }
```

- [ ] **Step 7: Rewrite `_calc_*` methods in `core.py` — new computation chain**

Replace the old velocity-based chain with the momentum-based chain. Remove `_calc_px`, `_calc_py`, `_calc_pz` (no longer derived). Add `_calc_vx`, `_calc_vy`, `_calc_vz`. Rewrite `_calc_p_abs`, `_calc_p_abs_si`, `_calc_gamma`, `_calc_beta`, `_calc_speed`.

Also remove the `_momentum_evc_backup` method and simplify `momentum_si` and `momentum_evc`.

```python
# --- Core momentum-based computation chain ---

def _calc_p_abs(self) -> np.ndarray:
    return np.sqrt(self._quantities["px"].data**2
                   + self._quantities["py"].data**2
                   + self._quantities["pz"].data**2)

def _calc_p_abs_si(self) -> np.ndarray:
    return self.p_abs * (abs(g_e0) / g_c)

def _calc_gamma(self) -> np.ndarray:
    p_si = self.p_abs_si
    return np.sqrt(1.0 + (p_si / (g_m0 * g_c)) ** 2)

def _calc_beta(self) -> np.ndarray:
    g = self.gamma
    return np.sqrt(1.0 - 1.0 / g**2)

def _calc_speed(self) -> np.ndarray:
    return self.beta * g_c

# --- Derived velocity components ---

def _calc_vx(self) -> np.ndarray:
    from .utils import _momentum_evc_components_to_velocity
    vx, _, _ = _momentum_evc_components_to_velocity(
        self._quantities["px"].data,
        self._quantities["py"].data,
        self._quantities["pz"].data,
    )
    return vx

def _calc_vy(self) -> np.ndarray:
    from .utils import _momentum_evc_components_to_velocity
    _, vy, _ = _momentum_evc_components_to_velocity(
        self._quantities["px"].data,
        self._quantities["py"].data,
        self._quantities["pz"].data,
    )
    return vy

def _calc_vz(self) -> np.ndarray:
    from .utils import _momentum_evc_components_to_velocity
    _, _, vz = _momentum_evc_components_to_velocity(
        self._quantities["px"].data,
        self._quantities["py"].data,
        self._quantities["pz"].data,
    )
    return vz

# --- Geometry (unchanged logic, now use derived vx/vy/vz) ---

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

# --- Angles ---

def _calc_xp(self) -> np.ndarray:
    return self._quantities["px"].data / self.p_abs

def _calc_yp(self) -> np.ndarray:
    return self._quantities["py"].data / self.p_abs

# --- Energy / charge ---

def _calc_Q_abs(self) -> np.ndarray:
    return np.abs(self.Q)

def _calc_kinetic_energy(self) -> np.ndarray:
    return (self.gamma - 1.0) * (g_m0 * g_c**2)

def _calc_kinetic_energy_eV(self) -> np.ndarray:
    return self.kinetic_energy / abs(g_e0)

# --- Current flux (unchanged logic, use derived vx/vy/vz) ---

def _calc_current_x(self) -> np.ndarray:
    return self.Q * self.vx

def _calc_current_y(self) -> np.ndarray:
    return self.Q * self.vy

def _calc_current_z(self) -> np.ndarray:
    return self.Q * self.vz

def _calc_current_abs(self) -> np.ndarray:
    return np.sqrt(self.current_flux_x**2 + self.current_flux_y**2 + self.current_flux_z**2)

def _calc_current_x_abs(self) -> np.ndarray:
    return np.abs(self.current_flux_x)

def _calc_current_y_abs(self) -> np.ndarray:
    return np.abs(self.current_flux_y)

def _calc_current_z_abs(self) -> np.ndarray:
    return np.abs(self.current_flux_z)
```

Also update `momentum_si` and `momentum_evc` (simplify — no gamma computation needed):

```python
def momentum_si(self, *, m0: float = g_m0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    factor = abs(g_e0) / g_c  # eV/c -> kg·m/s
    px_data = self._quantities["px"].data
    py_data = self._quantities["py"].data
    pz_data = self._quantities["pz"].data
    return px_data * factor, py_data * factor, pz_data * factor

def momentum_evc(self, *, m0: float = g_m0, e0: float = g_e0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    px_data = self._quantities["px"].data
    py_data = self._quantities["py"].data
    pz_data = self._quantities["pz"].data
    return px_data.copy(), py_data.copy(), pz_data.copy()
```

Delete the method `_momentum_evc_backup` entirely.

- [ ] **Step 8: Update property accessors in `core.py`**

Replace the `vx/vy/vz` direct-storage properties with computed ones, and make `px/py/pz` direct-storage properties:

```python
# Direct storage reads (base quantities)
@property
def px(self) -> np.ndarray:
    return self._quantities["px"].data

@property
def py(self) -> np.ndarray:
    return self._quantities["py"].data

@property
def pz(self) -> np.ndarray:
    return self._quantities["pz"].data

# Derived (computed from stored px/py/pz)
@property
def vx(self) -> np.ndarray:
    return self._calc_vx()

@property
def vy(self) -> np.ndarray:
    return self._calc_vy()

@property
def vz(self) -> np.ndarray:
    return self._calc_vz()
```

Also update the class docstring to reflect the new stored quantities:

```python
"""
Fully object-based particle distribution container.

Internally, all quantities are stored as ParticleArrayQuantity objects.
Externally, convenience properties such as .x, .px, .Q, .vz, .gamma
are preserved and return only the underlying ndarray.

Fixed stored quantities
-----------------------
x, y, z   : position [m]
px, py, pz: momentum [eV/c]
t         : time [s]
Q         : macro-particle charge [C]

Optional extras
---------------
Any other particle-resolved quantities are stored as extra quantities.
"""
```

- [ ] **Step 9: Run `test_core3d.py` to verify it works**

```bash
cd /home/duoxup/git_agent/partdist
python tests/test_core3d.py
```

Expected: no errors; output shows `particle_quantity_keys`, `mean("z")`, `std("z")`, `pz` array, `kinetic_energy_eV` array, `id` array.

- [ ] **Step 10: Commit**

```bash
git add src/partdist/pd3d/core.py tests/test_core3d.py
git commit -m "refactor(core): store px/py/pz as base, derive vx/vy/vz, improve gamma chain"
```

---

## Task 2: utils.py — Rename Helper, Simplify `_replace_velocity_from_momentum`

**Files:**
- Modify: `src/partdist/pd3d/utils.py`

- [ ] **Step 1: Rename `_update_velocity_components` → `_update_momentum_components` and update body**

```python
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
```

- [ ] **Step 2: Simplify `_replace_velocity_from_momentum`**

The function kept its original name (called from manipulator.py), but now simply delegates:

```python
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
```

- [ ] **Step 3: Update `_get_vxyz_data` docstring**

```python
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
```

- [ ] **Step 4: Run test_core3d.py again to confirm utils change doesn't break anything**

```bash
cd /home/duoxup/git_agent/partdist
python tests/test_core3d.py
```

Expected: same output as after Task 1.

- [ ] **Step 5: Commit**

```bash
git add src/partdist/pd3d/utils.py
git commit -m "refactor(utils): rename _update_velocity_components to _update_momentum_components, simplify _replace_velocity_from_momentum"
```

---

## Task 3: io.py — Remove Redundant Velocity↔Momentum Conversions

**Files:**
- Modify: `src/partdist/pd3d/io.py`

- [ ] **Step 1: Update `_reference_to_distribution` to use `px/py/pz` directly**

```python
def _reference_to_distribution(
    ref: AstraReferenceParticle,
    *,
    m0: float,
    species_key: str,
    status_key: str,
) -> ParticleDistribution:
    return ParticleDistribution(
        x=np.array([ref.x], dtype=float),
        y=np.array([ref.y], dtype=float),
        z=np.array([ref.z], dtype=float),
        px=np.array([ref.px], dtype=float),
        py=np.array([ref.py], dtype=float),
        pz=np.array([ref.pz], dtype=float),
        t=np.array([ref.t], dtype=float),
        Q=np.array([ref.Q], dtype=float),
        extras={
            species_key: ParticleArrayQuantity(
                name=species_key, data=np.array([ref.species], dtype=np.int64),
                unit="", dtype_kind="int", short_name=species_key,
                long_name="particle species flag", latex_name=species_key,
                category="flag", is_derived=False,
            ),
            status_key: ParticleArrayQuantity(
                name=status_key, data=np.array([ref.status], dtype=np.int64),
                unit="", dtype_kind="int", short_name=status_key,
                long_name="particle status flag", latex_name=status_key,
                category="flag", is_derived=False,
            ),
        },
    )
```

- [ ] **Step 2: Update `_concat_distributions` to use `px/py/pz`**

```python
def _concat_distributions(
    first: ParticleDistribution,
    second: ParticleDistribution,
) -> ParticleDistribution:
    if set(first.extra_quantity_keys) != set(second.extra_quantity_keys):
        raise ValueError("Both ParticleDistribution objects must have identical extra keys.")

    extras: dict[str, ParticleArrayQuantity] = {}
    for key in first.extra_quantity_keys:
        q1 = first.get_quantity(key)
        q2 = second.get_quantity(key)
        extras[key] = ParticleArrayQuantity(
            name=key, data=np.concatenate([q1.data, q2.data]),
            unit=q1.unit, dtype_kind=q1.dtype_kind,
            short_name=q1.short_name, long_name=q1.long_name,
            latex_name=q1.latex_name, category=q1.category,
            is_derived=False, is_discrete=q1.is_discrete,
            preferred_scale=q1.preferred_scale,
        )

    return ParticleDistribution(
        x=np.concatenate([first.x, second.x]),
        y=np.concatenate([first.y, second.y]),
        z=np.concatenate([first.z, second.z]),
        px=np.concatenate([first.px, second.px]),
        py=np.concatenate([first.py, second.py]),
        pz=np.concatenate([first.pz, second.pz]),
        t=np.concatenate([first.t, second.t]),
        Q=np.concatenate([first.Q, second.Q]),
        extras=extras,
    )
```

- [ ] **Step 3: Update `_build_reference_from_distribution` to use `px/py/pz` means directly**

```python
def _build_reference_from_distribution(
    dist: ParticleDistribution,
    *,
    m0: float = g_m0,
    mode: str = "mean",
    weight: str | np.ndarray | None = "absQ",
    reference_time: float = 0.0,
    reference_charge: float = 0.0,
    default_species: int = 1,
    default_status: int = 5,
    species_key: str = "species",
    status_key: str = "status",
) -> AstraReferenceParticle:
    """
    Build an ASTRA reference particle from ParticleDistribution.

    mode
    ----
    - "mean"  : use weighted means for x, y, z, px, py, pz
    - "zeros" : use zeros for x, y, z, px, py, pz
    """
    if mode == "mean":
        x0  = dist.mean("x",  weight=weight)
        y0  = dist.mean("y",  weight=weight)
        z0  = dist.mean("z",  weight=weight)
        px0 = dist.mean("px", weight=weight)
        py0 = dist.mean("py", weight=weight)
        pz0 = dist.mean("pz", weight=weight)
    elif mode == "zeros":
        x0 = y0 = z0 = px0 = py0 = pz0 = 0.0
    else:
        raise ValueError("mode must be 'mean' or 'zeros'.")

    return AstraReferenceParticle(
        x=float(x0), y=float(y0), z=float(z0),
        px=float(px0), py=float(py0), pz=float(pz0),
        t=float(reference_time),
        Q=float(reference_charge),
        species=int(default_species),
        status=int(default_status),
    )
```

- [ ] **Step 4: Update `read_astra_distribution` — pass `px/py/pz` directly**

In the `n == 0` branch, replace `vx/vy/vz` empty arrays with `px/py/pz`:

```python
real_dist = ParticleDistribution(
    x=np.empty(0, dtype=float),
    y=np.empty(0, dtype=float),
    z=np.empty(0, dtype=float),
    px=np.empty(0, dtype=float),
    py=np.empty(0, dtype=float),
    pz=np.empty(0, dtype=float),
    t=np.empty(0, dtype=float),
    Q=np.empty(0, dtype=float),
    extras={
        species_key: ParticleArrayQuantity(
            name=species_key, data=np.empty(0, dtype=np.int64),
            unit="", dtype_kind="int", short_name=species_key,
            long_name="particle species flag", latex_name=species_key,
            category="flag", is_derived=False,
        ),
        status_key: ParticleArrayQuantity(
            name=status_key, data=np.empty(0, dtype=np.int64),
            unit="", dtype_kind="int", short_name=status_key,
            long_name="particle status flag", latex_name=status_key,
            category="flag", is_derived=False,
        ),
    },
)
```

In the `else` branch (n > 0), remove the `momentum_evc_to_velocity` call and pass momentum directly:

```python
x  = data[:, 0] + ref.x
y  = data[:, 1] + ref.y
z  = data[:, 2] + ref.z

px = data[:, 3] + ref.px
py = data[:, 4] + ref.py
pz = data[:, 5] + ref.pz

t  = data[:, 6] * 1.0e-9 + ref.t
Q  = data[:, 7] * 1.0e-9

species = np.rint(data[:, 8]).astype(np.int64)
status  = np.rint(data[:, 9]).astype(np.int64)

real_dist = ParticleDistribution(
    x=x, y=y, z=z,
    px=px, py=py, pz=pz,
    t=t, Q=Q,
    extras={
        species_key: ParticleArrayQuantity(
            name=species_key, data=species, unit="", dtype_kind="int",
            short_name=species_key, long_name="particle species flag",
            latex_name=species_key, category="flag", is_derived=False,
        ),
        status_key: ParticleArrayQuantity(
            name=status_key, data=status, unit="", dtype_kind="int",
            short_name=status_key, long_name="particle status flag",
            latex_name=status_key, category="flag", is_derived=False,
        ),
    },
)
```

Also update the docstring comment for returned units from:
```
vx, vy, vz: m/s
```
to:
```
px, py, pz: eV/c
```

- [ ] **Step 5: Update `write_astra_distribution` — read `px/py/pz` from stored values**

Remove the `velocity_to_momentum_evc` call. Replace:

```python
# In the include_reference_particle branch: build ref from dist[0]
px0, py0, pz0 = velocity_to_momentum_evc(
    np.array([dist.vx[0]]),
    np.array([dist.vy[0]]),
    np.array([dist.vz[0]]),
    m0=m0,
)
ref = AstraReferenceParticle(
    x=float(dist.x[0]),   y=float(dist.y[0]),   z=float(dist.z[0]),
    px=float(px0[0]),      py=float(py0[0]),      pz=float(pz0[0]),
    t=float(dist.t[0]),   Q=float(dist.Q[0]),
    species=species0,      status=status0,
)
```

With:

```python
ref = AstraReferenceParticle(
    x=float(dist.x[0]),   y=float(dist.y[0]),   z=float(dist.z[0]),
    px=float(dist.px[0]), py=float(dist.py[0]), pz=float(dist.pz[0]),
    t=float(dist.t[0]),   Q=float(dist.Q[0]),
    species=species0,      status=status0,
)
```

And replace the momentum computation block before building `raw`:

```python
# Remove this block:
px, py, pz = velocity_to_momentum_evc(
    dist_particles.vx, dist_particles.vy, dist_particles.vz, m0=m0,
)

# Replace with:
px = dist_particles.px
py = dist_particles.py
pz = dist_particles.pz
```

- [ ] **Step 6: Run `test_core3d.py` to confirm io changes don't break the import chain**

```bash
cd /home/duoxup/git_agent/partdist
python tests/test_core3d.py
```

Expected: same output as before.

- [ ] **Step 7: Commit**

```bash
git add src/partdist/pd3d/io.py
git commit -m "refactor(io): remove redundant velocity<->momentum conversions on ASTRA read/write paths"
```

---

## Task 4: manipulator.py — Update Slope Space, Simplify `_update_quantity_array`, Fix `replicate_longitudinally`

**Files:**
- Modify: `src/partdist/pd3d/manipulator.py`

- [ ] **Step 1: Update import in `manipulator.py`**

Change the import of `_update_velocity_components` to `_update_momentum_components`:

```python
from .utils import (
    _as_1d_array,
    _copy_or_inplace,
    _extract_data,
    _get_weight_array,
    _normalize_mask,
    _update_momentum_components,
)
```

- [ ] **Step 2: Simplify `_update_quantity_array`**

`px/py/pz` are now base quantities and fall through the standard path. Remove the special derived-momentum branch entirely:

```python
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
```

- [ ] **Step 3: Update `match_twiss_plane` to use `px/pz` slope**

Find the `match_twiss_plane` function (around line 985). Update these sections:

The docstring definition line:
```python
# Before
- x plane: u = x,  u' = vx / vz
- y plane: u = y,  u' = vy / vz

# After
- x plane: u = x,  u' = px / pz
- y plane: u = y,  u' = py / pz
```

The data extraction block (replace `vx/vy/vz` reads with `px/py/pz`):

```python
if plane == "x":
    u  = _extract_data(out, "x",  n_expected=n, dtype=float, name="x")
    pp = _extract_data(out, "px", n_expected=n, dtype=float, name="px")
else:
    u  = _extract_data(out, "y",  n_expected=n, dtype=float, name="y")
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
```

The slope computation (variable renamed from `vp/vz` to `pp/pz`):

```python
up = pp / pz

u_sel  = u[valid]
up_sel = up[valid]
pz_sel = pz[valid]
w_sel  = w[valid]
```

The write-back block at the end:

```python
u_new  = u.copy()
pp_new = pp.copy()

u_new[valid]  = u_new_sel
pp_new[valid] = up_new_sel * pz_sel

if plane == "x":
    out.update_quantity("x", u_new)
    return _update_momentum_components(out, pp_new, out.py, out.pz, inplace=True)

out.update_quantity("y", u_new)
return _update_momentum_components(out, out.px, pp_new, out.pz, inplace=True)
```

- [ ] **Step 4: Update `replicate_longitudinally` base keys and construction call**

Find the `replicate_longitudinally` function (around line 1239). Update `base_keys`:

```python
base_keys = ("x", "y", "z", "px", "py", "pz", "t", "Q")
```

Update the `ParticleDistribution.from_arrays` call at the end of the function:

```python
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
```

- [ ] **Step 5: Run `test_core3d.py` to confirm nothing is broken**

```bash
cd /home/duoxup/git_agent/partdist
python tests/test_core3d.py
```

Expected: same output as before.

- [ ] **Step 6: Commit**

```bash
git add src/partdist/pd3d/manipulator.py
git commit -m "refactor(manipulator): use px/pz slope in match_twiss_plane, simplify _update_quantity_array, update replicate_longitudinally"
```
