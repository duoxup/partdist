"""Regression test: to_ocelot_particle_array rejects zero-reference-momentum input."""
import numpy as np
import pytest

from partdist import ParticleDistribution3D


def test_to_ocelot_rejects_zero_kinetic_energy():
    """A distribution with all particles at rest must raise, not produce inf/nan."""
    n = 64
    rest_only = ParticleDistribution3D(
        x=np.zeros(n), y=np.zeros(n), z=np.zeros(n),
        px=np.zeros(n), py=np.zeros(n), pz=np.zeros(n),
        t=np.zeros(n), Q=np.full(n, -1.6e-19),
    )
    # to_ocelot_particle_array may be exported from the package top-level. If not,
    # import from partdist.pd3d.io.
    try:
        from partdist import to_ocelot_particle_array
    except ImportError:
        from partdist.pd3d.io import to_ocelot_particle_array

    with pytest.raises(ValueError, match="reference momentum"):
        to_ocelot_particle_array(rest_only)
