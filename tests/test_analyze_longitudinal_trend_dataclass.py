"""Regression test: analyze_longitudinal_trend returns a frozen dataclass."""
from dataclasses import is_dataclass, fields
import numpy as np

from partdist import ParticleDistribution3D
from partdist.pd3d.analysis import analyze_longitudinal_trend


def _make_dist(n=2000):
    rng = np.random.default_rng(7)
    z = rng.normal(0, 1e-3, n)
    return ParticleDistribution3D(
        x=np.zeros(n), y=np.zeros(n), z=z,
        px=np.zeros(n), py=np.zeros(n),
        pz=1e7 + 5e5 * (z / 1e-3) + rng.normal(0, 1e4, n),
        t=np.zeros(n), Q=np.full(n, -1.6e-19),
    )


def test_returns_dataclass_with_three_fields():
    d = _make_dist()
    res = analyze_longitudinal_trend(d)
    assert is_dataclass(res), f"expected dataclass, got {type(res)!r}"
    field_names = {f.name for f in fields(res)}
    assert field_names == {"profile", "trend", "residuals"}, field_names


def test_fields_are_populated():
    d = _make_dist()
    res = analyze_longitudinal_trend(d)
    assert res.profile is not None
    assert res.trend is not None
    assert res.residuals is not None
