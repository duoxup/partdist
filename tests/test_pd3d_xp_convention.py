"""Regression tests pinning x' = px/pz across core and analysis modules."""
import numpy as np
from partdist import ParticleDistribution3D
from partdist.pd3d.analysis import compute_phase_space_plane


def _make_dist_with_transverse_kick():
    """1000-particle beam with deliberately non-paraxial transverse momentum.

    px, py have magnitudes comparable to the spread of pz, so px/|p| and
    px/pz disagree by a measurable factor.
    """
    rng = np.random.default_rng(0)
    n = 1000
    x = rng.normal(0.0, 1e-4, n)
    y = rng.normal(0.0, 1e-4, n)
    z = np.zeros(n)
    px = rng.normal(0.0, 1e5, n)
    py = rng.normal(0.0, 1e5, n)
    pz = 1e7 + rng.normal(0.0, 1e5, n)
    t = np.zeros(n)
    Q = np.full(n, -1.6e-19)
    return ParticleDistribution3D(
        x=x, y=y, z=z, px=px, py=py, pz=pz, t=t, Q=Q,
    )


def test_xp_equals_px_over_pz():
    d = _make_dist_with_transverse_kick()
    expected = d.get_data("px") / d.get_data("pz")
    np.testing.assert_allclose(d.get_data("xp"), expected, rtol=1e-12)


def test_yp_equals_py_over_pz():
    d = _make_dist_with_transverse_kick()
    expected = d.get_data("py") / d.get_data("pz")
    np.testing.assert_allclose(d.get_data("yp"), expected, rtol=1e-12)


def test_core_emit_x_matches_analysis_module():
    """core.emit_x and analysis.compute_phase_space_plane(plane='x').geometric_emittance must agree."""
    d = _make_dist_with_transverse_kick()
    from_core = d.emit_x
    from_analysis = compute_phase_space_plane(d, plane="x", weight="Q_abs").geometric_emittance
    np.testing.assert_allclose(from_core, from_analysis, rtol=1e-10)


def test_core_emit_y_matches_analysis_module():
    d = _make_dist_with_transverse_kick()
    from_core = d.emit_y
    from_analysis = compute_phase_space_plane(d, plane="y", weight="Q_abs").geometric_emittance
    np.testing.assert_allclose(from_core, from_analysis, rtol=1e-10)
