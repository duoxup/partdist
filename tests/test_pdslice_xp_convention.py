"""Regression tests pinning x' = px/pz for SliceDistribution (matches pd3d)."""
import numpy as np
from partdist import SliceDistribution
from partdist.pd3d.analysis import compute_phase_space_plane


def _make_slice_with_transverse_kick():
    """A 1000-particle slice with non-paraxial transverse momentum.

    px and pz have spreads of comparable order so px/|p| and px/pz disagree
    by a measurable factor.
    """
    rng = np.random.default_rng(0)
    n = 1000
    x = rng.normal(0.0, 1e-4, n)
    y = rng.normal(0.0, 1e-4, n)
    px = rng.normal(0.0, 1e5, n)
    py = rng.normal(0.0, 1e5, n)
    pz = 1e7 + rng.normal(0.0, 1e5, n)
    t = np.zeros(n)
    lam = np.full(n, 1e-9)  # arbitrary positive line-charge density
    return SliceDistribution(
        z=0.0, x=x, y=y, px=px, py=py, pz=pz, t=t, lam=lam,
    )


def test_slice_xp_equals_px_over_pz():
    s = _make_slice_with_transverse_kick()
    expected = s.get_data("px") / s.get_data("pz")
    np.testing.assert_allclose(s.get_data("xp"), expected, rtol=1e-12)


def test_slice_yp_equals_py_over_pz():
    s = _make_slice_with_transverse_kick()
    expected = s.get_data("py") / s.get_data("pz")
    np.testing.assert_allclose(s.get_data("yp"), expected, rtol=1e-12)


def test_slice_core_emit_x_matches_analysis_module():
    """SliceDistribution.emit_x and analysis.compute_phase_space_plane agree."""
    s = _make_slice_with_transverse_kick()
    from_core = s.emit_x
    from_analysis = compute_phase_space_plane(s, plane="x", weight="lam_abs").geometric_emittance
    np.testing.assert_allclose(from_core, from_analysis, rtol=1e-10)


def test_slice_core_emit_y_matches_analysis_module():
    s = _make_slice_with_transverse_kick()
    from_core = s.emit_y
    from_analysis = compute_phase_space_plane(s, plane="y", weight="lam_abs").geometric_emittance
    np.testing.assert_allclose(from_core, from_analysis, rtol=1e-10)
