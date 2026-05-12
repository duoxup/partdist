"""
partdist — particle distribution containers and I/O.

Top-level re-exports are intentionally small: container classes and I/O
entry points (the surface a user hits when first using the package).
Operations that are specific to one container type live under the
matching submodule:

- Manipulation of ``ParticleDistribution3D`` — ``partdist.pd3d.manipulator``
  (centering, masking, slicing, chirp/energy scaling, Twiss matching, …)
- Analysis (Twiss/emittance/current profile/binned fits) — ``partdist.pd3d.analysis``
- Visualisation — ``partdist.pd3d.viz``
- Low-level helpers (``momentum_evc_to_velocity``, …) — ``partdist.pd3d.utils``

Import the deeper functionality directly from those modules.
"""

from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version("partdist")
except PackageNotFoundError:
    __version__ = "0.0.0.dev"

from .particle_array_quantity import ParticleArrayQuantity
from .pd3d.core import ParticleDistribution3D, ParticleDistribution
from .pdslice.core import SliceDistribution
from .pd3d.io import (
    read_astra_distribution,
    write_astra_distribution,
    read_genesis_distribution,
    write_genesis_distribution,
    from_ocelot_particle_array,
    to_ocelot_particle_array,
)
from .pdslice.io import read_cst_pid_distribution


__all__ = [
    # Container types
    'ParticleArrayQuantity',
    'ParticleDistribution3D',
    'ParticleDistribution',          # backward-compatible alias
    'SliceDistribution',
    # I/O — file readers/writers and external-library bridges
    'read_astra_distribution',
    'write_astra_distribution',
    'read_genesis_distribution',
    'write_genesis_distribution',
    'read_cst_pid_distribution',
    'from_ocelot_particle_array',
    'to_ocelot_particle_array',
]
