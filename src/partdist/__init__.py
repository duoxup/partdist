"""
partdist — particle distribution containers, I/O, analysis, and visualisation.

Top-level re-exports cover the stable surface. Less-frequently-used
manipulator routines (centering, masking, energy/chirp scaling, ...)
live under `partdist.pd3d.manipulator`; import them from there directly.
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
    from_ocelot_particle_array,
    to_ocelot_particle_array,
)
from .pdslice.io import read_cst_pid_distribution

from .pd3d.manipulator import (
    replicate_longitudinally,
    multiply_longitudinal_profile,
    set_linear_chirp,
    match_twiss_xy,
                               )


__all__ = [
    'ParticleArrayQuantity',
    'ParticleDistribution3D',
    'ParticleDistribution',  # backward-compatible alias
    'SliceDistribution',
    'read_astra_distribution',
    'write_astra_distribution',
    'read_cst_pid_distribution',
    'from_ocelot_particle_array',
    'to_ocelot_particle_array',
    'replicate_longitudinally',
    'multiply_longitudinal_profile',
    'set_linear_chirp',
    'match_twiss_xy',
    
    
    
    ]