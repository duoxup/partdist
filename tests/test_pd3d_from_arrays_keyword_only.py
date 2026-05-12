"""Regression test: pd3d.ParticleDistribution3D.from_arrays is keyword-only, matching pdslice."""
import inspect

import numpy as np
import pytest

from partdist import ParticleDistribution3D


def test_from_arrays_all_params_keyword_only():
    sig = inspect.signature(ParticleDistribution3D.from_arrays)
    for name, param in sig.parameters.items():
        if name == "cls":
            continue
        assert param.kind == inspect.Parameter.KEYWORD_ONLY, (
            f"parameter {name!r} kind is {param.kind!r}, expected KEYWORD_ONLY"
        )


def test_from_arrays_positional_call_rejected():
    n = 10
    args = [np.zeros(n)] * 7  # x, y, z, px, py, pz, t — 7 positional
    with pytest.raises(TypeError):
        ParticleDistribution3D.from_arrays(*args)


def test_from_arrays_keyword_call_works():
    n = 10
    d = ParticleDistribution3D.from_arrays(
        x=np.zeros(n), y=np.zeros(n), z=np.zeros(n),
        px=np.zeros(n), py=np.zeros(n), pz=np.full(n, 1e7),
        t=np.zeros(n),
    )
    assert len(d) == n
