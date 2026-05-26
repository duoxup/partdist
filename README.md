# PartDist: 3D Particle Distribution Storage and Manipulation

PartDist is a Python library for storing, manipulating, and analyzing charged-particle distributions in accelerator-physics simulations. It exposes two object-oriented containers — `ParticleDistribution3D` for full 3D point clouds and `SliceDistribution` for steady-state distributions on a single transverse plane — and bridges them to common simulation formats (ASTRA, Genesis, CST Particle Studio `.pid`, OCELOT).

## Quick Start

### Installation

```bash
pip install git+https://github.com/duoxup/partdist.git
```

### Minimal Example

```python
import numpy as np
from partdist import ParticleDistribution3D
from partdist.pd3d.utils import velocity_to_momentum_evc

# Sample particle data
x = np.array([0.0, 1e-6, -1e-6])
y = np.array([0.0, 2e-6, -2e-6])
z = np.array([0.0, 1e-3, -1e-3])

vx = np.array([0.0, 1e5, -1e5])
vy = np.array([0.0, 2e5, -2e5])
vz = np.array([2.9e8, 2.9e8, 2.9e8])

# Convert velocity to momentum (eV/c)
px, py, pz = velocity_to_momentum_evc(vx, vy, vz)

t = np.array([0.0, 3e-12, -3e-12])
Q = np.array([-1e-12, -1e-12, -1e-12])

# Construct the distribution (note: from_arrays is keyword-only)
dist = ParticleDistribution3D.from_arrays(
    x=x, y=y, z=z,
    px=px, py=py, pz=pz,
    t=t, Q=Q,
)

print("Particle count:", len(dist))
print("Mean z:", dist.mean("z"))
print("Standard deviation of z:", dist.std("z"))
print("Momentum z-component:", dist.pz)
print("Kinetic energy (eV):", dist.kinetic_energy_eV)
```

The legacy name `ParticleDistribution` is preserved as an alias for `ParticleDistribution3D` and will continue to work; new code should prefer the canonical name.

## Core Concepts

### Storage Model

`ParticleDistribution3D` stores the following **base quantities** directly:
- `x, y, z`: Position coordinates [m]
- `px, py, pz`: Momentum components [eV/c]
- `t`: Time [s]
- `Q`: Macro-particle charge [C, signed]

The following **derived quantities** are computed on-demand from the base columns (and cached where the cost matters):
- `vx, vy, vz`: Velocity components [m/s]
- `gamma, beta, speed`: Relativistic factors and total speed
- `p_abs, p_abs_si`: Momentum magnitude in eV/c and SI
- `kinetic_energy, kinetic_energy_eV`: Kinetic energy in J and eV
- `xp, yp`: Geometric divergences `px/pz`, `py/pz` [rad]
- `delta`: Relative momentum deviation `(|p| − ⟨|p|⟩) / ⟨|p|⟩`
- `tau`: Longitudinal position offset `z_ref − z` [m]
- `Q_abs`: Absolute macro-particle charge [C]
- `radial_position, transverse_speed, radial_velocity, azimuthal_velocity`
- `current_flux_x/y/z` and absolute variants

### `SliceDistribution`

`SliceDistribution` is the slice-plane analogue of `ParticleDistribution3D`. All particles live on a single z-plane (`z` is a scalar); the longitudinal observable is `lam` (linear charge density along z, C/m) instead of a per-particle `Q`. It is the natural container for CST `.pid` emission planes and other steady-state cross-sections. Same derived-quantity machinery as the 3D container; transverse Twiss diagnostics work the same way.

### Unit System

- Positions: meters [m]
- Momentum: electron-volts per speed of light [eV/c]
- Time: seconds [s]
- Charge: Coulombs [C]
- Line charge density (slice): Coulombs per metre [C/m]

`partdist.kinematics` provides relativistic conversions; `partdist.pd3d.utils` provides the canonical `(p_eVc) ↔ (v_si)` converters.

## Detailed Examples

### Reading and Writing ASTRA Files

```python
from partdist import read_astra_distribution, write_astra_distribution

dist = read_astra_distribution("input.dist")

print(f"Loaded {len(dist)} particles")
print(f"Mean energy: {dist.mean('kinetic_energy_eV', weight='Q_abs'):.2f} eV")

write_astra_distribution("output.dist", dist)
```

### Reading a CST `.pid` Emission Plane

```python
from partdist import read_cst_pid_distribution

# Coplanar emission points (one z within plane_tol) become a SliceDistribution
slice_dist = read_cst_pid_distribution("emission.pid")
print(f"slice z₀ = {slice_dist.z:.6g} m, particles = {len(slice_dist)}")
print(f"total beam current = {slice_dist.lam.sum() * slice_dist.mean('vz'):.4g} A (approx)")
```

A `.pid` file whose points are not coplanar within `plane_tol` (default `1e-9 m`) is rejected — load it as a `ParticleDistribution3D` instead.

### OCELOT Bridge

```python
from partdist import from_ocelot_particle_array, to_ocelot_particle_array
from ocelot.cpbd.beam import ParticleArray  # external

pa = ParticleArray(n=10000)
# ... ocelot populates pa.rparticles, pa.q_array, pa.E, pa.s ...

dist = from_ocelot_particle_array(pa)
# ... analyse / manipulate in partdist ...
pa_back = to_ocelot_particle_array(dist)
```

`to_ocelot_particle_array` raises `ValueError` if the input distribution has no usable reference momentum (e.g. all particles at rest); this is intentional, because the OCELOT phase-space coordinates `xp = px/p₀c` are undefined for `p₀c = 0`.

### Matching Twiss Parameters

```python
from partdist.pd3d.manipulator import match_twiss_xy
from partdist.pd3d.analysis import compute_phase_space_plane

dist_matched = match_twiss_xy(
    dist,
    alpha_x=0.0,
    beta_x=0.2,           # m
    alpha_y=-1.0,
    beta_y=0.2,
    weight="Q_abs",
    center_before_match=True,
    preserve_centroid=True,
)

twiss_x = compute_phase_space_plane(dist_matched, plane="x", weight="Q_abs")
twiss_y = compute_phase_space_plane(dist_matched, plane="y", weight="Q_abs")

print(f"alpha_x = {twiss_x.alpha:.3f}, beta_x = {twiss_x.beta:.3f} m, eps_geom = {twiss_x.geometric_emittance:.3e} m")
print(f"alpha_y = {twiss_y.alpha:.3f}, beta_y = {twiss_y.beta:.3f} m, eps_geom = {twiss_y.geometric_emittance:.3e} m")
```

The package standardises on the geometric divergence `x' = px/pz`, `y' = py/pz` (the textbook accelerator-physics convention); both `ParticleDistribution3D.emit_x/y` and `compute_phase_space_plane(...).geometric_emittance` use this definition.

### Longitudinal Manipulation

```python
from partdist.pd3d.manipulator import (
    replicate_longitudinally,
    multiply_longitudinal_profile,
    set_linear_chirp,
)

dist_replicated = replicate_longitudinally(
    dist,
    n_copies=5,
    spacing=3e-4,         # 0.3 mm spacing
    sort_by=None,
)

def parabolic_profile(z_max):
    return lambda z: 3 / (4 * z_max) * (1 - z ** 2 / z_max ** 2)

dist_rescaled = multiply_longitudinal_profile(
    dist_replicated,
    profile_func=parabolic_profile(9e-4),
    center="mean",
    normalize=True,
)

chirp_rate = -200e3 / dist_rescaled.std("z")    # −200 keV per σ_z
dist_chirped = set_linear_chirp(
    dist_rescaled,
    slope=chirp_rate,
    center_x=True,
    center_y=True,
    preserve_mean_kinetic_energy=True,
    weight_for_energy="Q",
)
```

### Analysis and Visualization

```python
from partdist.pd3d.analysis import current_profile_z, compute_phase_space_plane, analyze_longitudinal_trend
from partdist.pd3d.viz import hist2d_pd3d
import matplotlib.pyplot as plt

# Longitudinal current profile
z_bins, current = current_profile_z(dist, bins=100)
plt.figure()
plt.plot(z_bins, current)
plt.xlabel("z [m]")
plt.ylabel("Current [A]")
plt.title("Longitudinal Current Profile")

# Phase-space heat map
fig, ax, *_ = hist2d_pd3d(
    dist,
    x="z", y="pz",
    color_threshold=1e-2,
    cmap="jet",
)
ax.set_xlabel("z [m]")
ax.set_ylabel("pz [eV/c]")
ax.set_title("Phase Space Distribution")

# Twiss
twiss = compute_phase_space_plane(dist, plane="x", weight="Q_abs")
print(f"eps_geom = {twiss.geometric_emittance:.3e} m, alpha_x = {twiss.alpha:.3f}, beta_x = {twiss.beta:.3f} m")

# pz(z) trend + residuals as a single dataclass result
trend = analyze_longitudinal_trend(dist)
print("profile bin count:", len(trend.profile.x_centers))
print("trend method:", trend.trend.method)
```

Composite analysis results are frozen dataclasses (`PhaseSpacePlaneResult`, `BinnedProfileResult`, `TrendFitResult`, `AnalyzeLongitudinalTrendResult`, `BeamDiagnosticsResult`, …) — access by attribute, not by dict key.

## API Overview

### Main Classes

- `ParticleDistribution3D` — full 3D particle distribution container
- `SliceDistribution` — single-plane (slice) distribution; `lam` instead of `Q`
- `ParticleArrayQuantity` — individual array column with unit/name/category metadata
- `ParticleDistribution` — backward-compatibility alias of `ParticleDistribution3D`

### File I/O (top-level re-exports)

- `read_astra_distribution()`, `write_astra_distribution()` — ASTRA `.ini` / `.dist`
- `read_genesis_distribution()`, `write_genesis_distribution()` — Genesis 4 HDF5
- `read_cst_pid_distribution()` — CST Particle Studio `.pid` emission planes → `SliceDistribution`
- `from_ocelot_particle_array()`, `to_ocelot_particle_array()` — OCELOT `ParticleArray` bridging

Low-level relativistic helpers live in `partdist.pd3d.utils`:
- `momentum_evc_to_velocity()`, `velocity_to_momentum_evc()` — `(p_eVc) ↔ (v_si)` converters

### Manipulation (in `partdist.pd3d.manipulator`)

These functions operate on `ParticleDistribution3D` specifically; import them from `partdist.pd3d.manipulator` rather than the top-level package.

Headline routines:
- `replicate_longitudinally()` — create longitudinal copies of a distribution
- `multiply_longitudinal_profile()` — re-weight by a user-supplied profile function
- `set_linear_chirp()` — impose a `δ(z)` linear energy chirp
- `match_twiss_xy()` — match transverse Twiss parameters in both planes

Plus a larger set of helpers (centering, masking, slicing, core-region extraction, `scale_emittance`, `scale_energy`, …) — see the module for the full surface.

### Analysis (in `partdist.pd3d.analysis`)

- `compute_phase_space_plane()` — full Twiss + geometric/normalised emittance for one plane (and `compute_phase_space_covariance_plane()` for just the 2×2 covariance matrix)
- `current_profile_z()` — longitudinal current profile via charge-weighted histogram
- `compute_binned_profile()`, `fit_trend_from_profile()`, `evaluate_residuals()` — binned-statistic pipeline
- `analyze_longitudinal_trend()` — convenience wrapper: returns `AnalyzeLongitudinalTrendResult` bundling profile + trend + residuals
- `compute_longitudinal_linearity()`, `compute_beam_diagnostics()`, `fit_current_profile()` — higher-level diagnostics

Normalised emittance follows the textbook convention `eps_norm = β₀γ₀ · eps_geom` (β₀γ₀ from the charge-weighted reference momentum).

### Visualization (in `partdist.pd3d.viz`)

- `hist2d_pd3d()` — 2D phase-space histogram with automatic SI-prefix axis labels

## Contributing

### Development Setup

```bash
git clone https://github.com/duoxup/partdist.git
cd partdist
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest tests/ -v
```

All files under `tests/` are proper pytest tests (`def test_*()` functions with `assert` statements). A handful of them are skipped automatically when their HDF5 fixture data is not present.

Older script-style exploration files (top-level `os.chdir`, runtime side effects, no `assert`) live under `local/`, which is gitignored — they are not test fixtures and are not run by CI.

### Code Style

- Follow PEP 8.
- Use type hints for function signatures.
- Document public APIs with docstrings (NumPy style).
- Run `ruff check` and `black` before committing.


---

PartDist is designed for accelerator physicists and researchers working with particle-beam simulations. It bridges the gap between low-level particle data and higher-level beam-dynamics operations.
