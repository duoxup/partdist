from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version("partdist")
except PackageNotFoundError:
    __version__ = "0.0.0.dev"
    
# import sys as _sys 

# from . import pd3d

from .particle_array_quantity import ParticleArrayQuantity
from .pd3d.core import ParticleDistribution3D, ParticleDistribution
from .pdslice.core import SliceDistribution
from .pd3d.io import (
    read_astra_distribution,
    write_astra_distribution,
    read_cst_pid_distribution,
    from_ocelot_particle_array,
    to_ocelot_particle_array,
)

from .pd3d.manipulator import (
    replicate_longitudinally,
    multiply_longitudinal_profile,
    set_linear_chirp,
    match_twiss_xy,
                               )



# _sys.modules[__name__ + '.pd3d'] = pd3d


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