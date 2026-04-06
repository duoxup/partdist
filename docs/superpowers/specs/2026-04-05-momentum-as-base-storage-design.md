# Design: Switch Core Storage from Velocity to Momentum

**Date:** 2026-04-05  
**Status:** Approved  

## Background

`ParticleDistribution` currently stores `vx, vy, vz` (m/s) as base quantities and derives `px, py, pz` (eV/c) from them. For high-energy accelerator particles where β ≈ 1, computing γ from velocity suffers from catastrophic cancellation in `1 − β²`. Storing momentum directly is physically more natural and numerically more stable.

## Goal

Swap the canonical storage: `px, py, pz` (eV/c) become base quantities; `vx, vy, vz` become derived. Improve the γ/β computation chain accordingly. No backward-compatibility shims — clean break only.

---

## Section 1: Data Model (core.py)

### Base / Derived Reclassification

| Quantity | Old | New | Unit |
|----------|-----|-----|------|
| x, y, z  | base | base (unchanged) | m |
| **vx, vy, vz** | **base** | **derived** | m/s |
| **px, py, pz** | **derived** | **base** | **eV/c** |
| t, Q | base | base (unchanged) | s, C |

### New Derived Computation Chain

```
px/py/pz (stored)
  → p_abs    = sqrt(px² + py² + pz²)           [eV/c]
  → p_abs_si = p_abs * |e0| / c                [kg·m/s]
  → gamma    = sqrt(1 + p_abs_si² / (m0·c)²)  [dimensionless]
  → beta     = sqrt(1 − 1/γ²)                 [dimensionless]
  → speed    = beta · c                        [m/s]
  → vx/vy/vz = speed · (px/py/pz) / p_abs     [m/s]
```

This replaces the old chain `vx/vy/vz → speed → beta → gamma`, which was numerically poor for ultra-relativistic particles.

### Interface Changes

The following methods have their signatures updated from `vx/vy/vz` to `px/py/pz`:
- `__init__`
- `from_arrays` (classmethod)
- `from_dict` (classmethod) — required keys become `("x","y","z","px","py","pz","t","Q")`
- `copy()`
- `slice()`

Statistics helpers:
- `centroid()` — replace `vx/vy/vz` entries with `px/py/pz`
- `sigma_dict()` — same

Property accessors:
- `dist.px`, `dist.py`, `dist.pz` — become direct storage reads (no computation)
- `dist.vx`, `dist.vy`, `dist.vz` — become computed properties via `_calc_vx/vy/vz()`

Docstring updated: stored quantities are now `x, y, z, px, py, pz, t, Q`.

---

## Section 2: Utility Functions (utils.py)

### `_update_velocity_components` → `_update_momentum_components`

The function is renamed and updated to operate on the base quantities `px/py/pz` instead of `vx/vy/vz`, calling `dist.update_quantity("px"/"py"/"pz", ...)`.

### `_replace_velocity_from_momentum` Simplified

Old: convert momentum → velocity → update stored `vx/vy/vz` (two steps).  
New: directly delegates to `_update_momentum_components` (one step, zero conversion).  
Function is kept under the same name to minimize call-site changes in manipulator.py.

### `_get_pxyz_data` / `_get_vxyz_data`

- `_get_pxyz_data`: reads directly from stored base quantities — no computation
- `_get_vxyz_data`: reads from derived properties `dist.vx/vy/vz` — still valid

Signatures and return types unchanged.

---

## Section 3: IO (io.py)

### `read_astra_distribution` Simplified

ASTRA files natively store momentum in eV/c. The current `momentum_evc_to_velocity` call on the read path is eliminated — `px/py/pz` are passed directly to `ParticleDistribution`:

```python
# Before
vx, vy, vz = momentum_evc_to_velocity(px, py, pz, m0=m0)
ParticleDistribution(x=x, ..., vx=vx, vy=vy, vz=vz, ...)

# After
ParticleDistribution(x=x, ..., px=px, py=py, pz=pz, ...)
```

### `write_astra_distribution` Simplified

The `velocity_to_momentum_evc` call on the write path is eliminated — `dist.px/py/pz` are read directly from storage:

```python
# Before
px, py, pz = velocity_to_momentum_evc(dist.vx, dist.vy, dist.vz, m0=m0)

# After
px, py, pz = dist.px, dist.py, dist.pz
```

### Other Internal Functions

- `_build_reference_from_distribution`: compute reference `px0/py0/pz0` by taking weighted means of stored `px/py/pz` directly, removing the intermediate `vx/vy/vz` mean + conversion step.
- `_reference_to_distribution`: construct `ParticleDistribution` with `px/py/pz` instead of `vx/vy/vz`.
- `_concat_distributions`: updated to pass `px/py/pz` to `ParticleDistribution`.

`momentum_evc_to_velocity` and `velocity_to_momentum_evc` module-level functions are **kept** — they are public conversion utilities.

---

## Section 4: Manipulator (manipulator.py)

### `match_twiss_plane` — Use px/pz Slope Directly

Since `vx/vz = px/pz` exactly (γ cancels), the transverse slope definition is unchanged. The implementation is updated to read `px/pz` from storage and write back through `_update_momentum_components`, avoiding any pass through derived velocity properties:

```python
# Before
up = vx / vz  (via derived properties)
vp_new[valid] = up_new_sel * vz_sel
_update_velocity_components(out, vp_new, out.vy, out.vz, inplace=True)

# After
up = px / pz  (direct storage reads)
pp_new[valid] = up_new_sel * pz_sel
_update_momentum_components(out, pp_new, out.py, out.pz, inplace=True)
```

Docstring updated: slope definition stated as `u' = px/pz` (or `py/pz`).

### `_update_quantity_array` Simplified

`px/py/pz` are now base quantities, so they follow the standard `update_quantity` path. The special derived-momentum branch (which previously converted to velocity and wrote back `vx/vy/vz`) is removed entirely:

```python
# Before: required a special branch for px/py/pz as derived quantities
if kind == "derived":
    if key in {"px", "py", "pz"}:
        return _replace_velocity_from_momentum(...)

# After: px/py/pz fall through the standard base/extra path
if kind in {"base", "extra"}:
    out.update_quantity(key, arr)
    return out
```

### `replicate_longitudinally`

`base_keys` updated from `("x","y","z","vx","vy","vz","t","Q")` to `("x","y","z","px","py","pz","t","Q")`. The `ParticleDistribution.from_arrays` call at the end updated accordingly.

### `_infer_reference_velocity`

Uses `dist.vz` — this remains a valid derived property after the change. **No modification needed.**

---

## Files Changed

| File | Nature of Change |
|------|-----------------|
| `src/partdist/pd3d/core.py` | Base/derived swap, new computation chain, signature updates |
| `src/partdist/pd3d/utils.py` | Rename `_update_velocity_components`, simplify `_replace_velocity_from_momentum` |
| `src/partdist/pd3d/io.py` | Remove redundant velocity↔momentum conversions on read/write paths |
| `src/partdist/pd3d/manipulator.py` | Use px/pz slope, simplify `_update_quantity_array`, update `replicate_longitudinally` |

## Files NOT Changed

| File | Reason |
|------|--------|
| `src/partdist/pd3d/analysis.py` | Accesses quantities via `dist.get_data(key)` — agnostic to storage |
| `src/partdist/pd3d/viz.py` | Same — no direct dependency on vx/vy/vz or px/py/pz as base |
| `src/partdist/particle_array_quantity.py` | Pure data container, unchanged |
| `src/partdist/__init__.py` | Public API surface unchanged |
| `tests/` | Test files will need updating to construct `ParticleDistribution` with `px/py/pz` |
