# PartDist: 3D Particle Distribution Storage and Manipulation

PartDist is a Python library for storing, manipulating, and analyzing 3D particle distributions in accelerator physics simulations. It provides an object-oriented interface to work with particle data, supporting operations like Twiss parameter matching, longitudinal manipulation, and ASTRA file I/O.

## Quick Start

### Installation

```bash
pip install git+https://github.com/PigDuo/partdist.git
```

### Minimal Example

```python
import numpy as np
from partdist import ParticleDistribution
from partdist.pd3d.io import velocity_to_momentum_evc

# Create sample particle data
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

# Create particle distribution
dist = ParticleDistribution.from_arrays(
    x=x, y=y, z=z,
    px=px, py=py, pz=pz,
    t=t, Q=Q
)

# Access properties
print("Particle count:", len(dist))
print("Mean z:", dist.mean("z"))
print("Standard deviation of z:", dist.std("z"))
print("Momentum z-component:", dist.pz)
print("Kinetic energy (eV):", dist.kinetic_energy_eV)
```

## Core Concepts

### Storage Model

`ParticleDistribution` stores the following **base quantities** directly:
- `x, y, z`: Position coordinates [m]
- `px, py, pz`: Momentum components [eV/c]
- `t`: Time [s]
- `Q`: Macro-particle charge [C]

The following **derived quantities** are computed on-demand:
- `vx, vy, vz`: Velocity components [m/s]
- `gamma, beta`: Lorentz factor and normalized velocity
- `p_abs`: Momentum magnitude [eV/c]
- `kinetic_energy_eV`: Kinetic energy [eV]
- And many more (transverse speed, radial position, current-like weights, etc.)

### Unit System

- Positions: meters [m]
- Momentum: electron-volts per speed of light [eV/c]
- Time: seconds [s]
- Charge: Coulombs [C]

The library uses `partdist.kinematics` for relativistic conversions and ensures consistent unit handling throughout all operations.

## Detailed Examples

### Reading and Writing ASTRA Files

```python
from partdist import read_astra_distribution, write_astra_distribution

# Read ASTRA distribution file
dist = read_astra_distribution("input.dist")

# Work with the distribution
print(f"Loaded {len(dist)} particles")
print(f"Mean energy: {dist.mean('kinetic_energy_eV', weight='absQ'):.2f} eV")

# Write back to ASTRA format
write_astra_distribution("output.dist", dist)
```

### Matching Twiss Parameters

```python
from partdist import match_twiss_xy

# Match both transverse planes
dist_matched = match_twiss_xy(
    dist,
    alpha_x=0.0,   # target alpha_x
    beta_x=0.2,    # target beta_x [m]
    alpha_y=-1.0,  # target alpha_y
    beta_y=0.2,    # target beta_y [m]
    weight='absQ',  # weight by absolute charge
    center_before_match=True,
    preserve_centroid=True
)

# Verify matched parameters
from partdist.pd3d.analysis import compute_twiss_plane
twiss_x = compute_twiss_plane(dist_matched, plane='x', weight='absQ')
twiss_y = compute_twiss_plane(dist_matched, plane='y', weight='absQ')

print(f"Matched alpha_x: {twiss_x.alpha:.3f}, beta_x: {twiss_x.beta:.3f} m")
print(f"Matched alpha_y: {twiss_y.alpha:.3f}, beta_y: {twiss_y.beta:.3f} m")
```

### Longitudinal Manipulation

```python
from partdist import replicate_longitudinally, multiply_longitudinal_profile, set_linear_chirp
import numpy as np

# Replicate distribution longitudinally
dist_replicated = replicate_longitudinally(
    dist, 
    n_copies=5, 
    spacing=3e-4,  # 0.3 mm spacing
    sort_by=None
)

# Apply longitudinal current profile
def parabolic_profile(z_max):
    """Return inverted parabola profile function"""
    return lambda z: 3 / (4*z_max) * (1 - z**2 / z_max**2)

dist_rescaled = multiply_longitudinal_profile(
    dist_replicated,
    profile_func=parabolic_profile(9e-4),  # 0.9 mm half-length
    center='mean',
    normalize=True
)

# Apply linear energy chirp
chirp_rate = -200e3 / dist_rescaled.std('z')  # -200 keV over 1 sigma_z
dist_chirped = set_linear_chirp(
    dist_rescaled,
    slope=chirp_rate,
    center_x=True,
    center_y=True,
    preserve_mean_kinetic_energy=True,
    weight_for_energy="Q"
)
```

### Analysis and Visualization

```python
from partdist.pd3d.analysis import current_profile_z, compute_twiss_plane
from partdist.pd3d.viz import hist2d_pd3d
import matplotlib.pyplot as plt

# Compute current profile along z
z_bins, current = current_profile_z(dist, bins=100)
plt.figure()
plt.plot(z_bins, current)
plt.xlabel('z [m]')
plt.ylabel('Current [A]')
plt.title('Longitudinal Current Profile')

# Create 2D histogram visualization
fig, ax, *_ = hist2d_pd3d(
    dist,
    x='z', y='pz',
    color_threshold=1e-2,
    cmap='jet'
)
ax.set_xlabel('z [m]')
ax.set_ylabel('pz [eV/c]')
ax.set_title('Phase Space Distribution')

# Compute emittance and Twiss parameters
twiss = compute_twiss_plane(dist, plane='x', weight='absQ')
print(f"ε_x = {twiss.geometric_emittance:.3e} m·rad, α_x = {twiss.alpha:.3f}, β_x = {twiss.beta:.3f} m")
```

## API Overview

### Main Classes

- `ParticleDistribution`: Core container for particle data
- `ParticleArrayQuantity`: Individual quantity with metadata (unit, name, category)

### Core Functions

**I/O Operations:**
- `read_astra_distribution()`: Read ASTRA particle distribution files
- `write_astra_distribution()`: Write to ASTRA format
- `read_genesis_distribution()`: Read GENESIS 4 `.h5` particle output files

**Manipulation:**
- `replicate_longitudinally()`: Create longitudinal copies
- `multiply_longitudinal_profile()`: Apply current profile
- `set_linear_chirp()`: Apply linear energy chirp
- `match_twiss_xy()`: Match transverse Twiss parameters

**Analysis (in `pd3d.analysis`):**
- `compute_twiss_plane()`: Compute Twiss parameters for a plane
- `current_profile_z()`: Compute longitudinal current profile

**Visualization (in `pd3d.viz`):**
- `hist2d_pd3d()`: Create 2D histogram visualizations

## Contributing

### Development Setup

```bash
git clone https://github.com/PigDuo/partdist.git
cd partdist

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/
```

### Code Style

- Follow PEP 8 conventions
- Use type hints for function signatures
- Document public APIs with docstrings
- Run `ruff check` and `black` before committing


---

PartDist is designed for accelerator physicists and researchers working with particle beam simulations. It bridges the gap between low-level particle data and high-level beam dynamics operations.
