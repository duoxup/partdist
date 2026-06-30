"""compute_beam_diagnostics exposes the longitudinal nonlinearity coefficient
(f_nonlinear) and keeps it consistent with compute_longitudinal_linearity."""
import numpy as np

from partdist import ParticleDistribution3D
from partdist.pd3d.analysis import (
    compute_beam_diagnostics,
    compute_longitudinal_linearity,
)


def _make_nonlinear_chirp_dist(n=5000):
    """Beam with a deliberately quadratic pz(z) so f_nonlinear > 0."""
    rng = np.random.default_rng(7)
    z = rng.normal(0.0, 1e-3, n)
    zn = z / 1e-3
    pz = 1e7 + 1e6 * zn + 5e5 * zn**2 + rng.normal(0.0, 1e4, n)
    return ParticleDistribution3D(
        x=rng.normal(0.0, 1e-4, n), y=rng.normal(0.0, 1e-4, n), z=z,
        px=rng.normal(0.0, 1e3, n), py=rng.normal(0.0, 1e3, n), pz=pz,
        t=np.zeros(n), Q=np.full(n, -1.6e-19),
    )


def test_beam_diagnostics_reports_f_nonlinear():
    d = _make_nonlinear_chirp_dist()
    res = compute_beam_diagnostics(d)

    expected = compute_longitudinal_linearity(d).f_nonlinear

    assert res.f_nonlinear == expected
    assert res.f_nonlinear > 0.0


def test_f_nonlinear_in_to_dict():
    d = _make_nonlinear_chirp_dist()
    res = compute_beam_diagnostics(d)
    assert "f_nonlinear" in res.to_dict()
