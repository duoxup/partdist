from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version("partdist")
except PackageNotFoundError:
    __version__ = "0.0.0.dev"
    
import sys as _sys 

from . import pd3d

from .particle_array_quantity import ParticleArrayQuantity
from .pd3d.core import ParticleDistribution
from .pd3d.io import read_astra_distribution, write_astra_distribution

from .pd3d.manipulator import (
    replicate_longitudinally,
    multiply_longitudinal_profile,
    set_linear_chirp,
    match_twiss_xy,
                               )



_sys.modules[__name__ + '.pd3d'] = pd3d


__all__ = [
    'ParticleArrayQuantity',
    'ParticleDistribution',
    'read_astra_distribution',
    'write_astra_distribution',
    'replicate_longitudinally',
    'multiply_longitudinal_profile',
    'set_linear_chirp',
    'match_twiss_xy',
    
    
    
    ]