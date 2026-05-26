"""Regression test: container `nemit_x/y` matches
`compute_phase_space_plane(...).normalized_emittance` for a wide-energy-spread beam.

Pre-fix the container used `beta0 * gamma0 * emit_*` (β₀, γ₀ derived from <E_k>),
while `compute_phase_space_plane` used `<|p|>/(m₀c) * emit_*`. The two diverge
under broad energy spread. After the fix both routes use `<|p|>/(m₀c)`.
"""
import numpy as np
from partdist import ParticleDistribution3D, SliceDistribution
from partdist.pd3d.analysis import compute_phase_space_plane


def _make_wide_spread_3d(n=5000, seed=4):
    """Mildly relativistic beam with broad pz spread, finite y and py.

    Mildly relativistic so that ``<βγ>`` (distribution average) and
    ``β(<E_k>)·γ(<E_k>)`` (single-particle from mean kinetic energy) differ
    by a measurable amount — at ultra-relativistic γ they coincide to
    leading order.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1e-4, n)
    y = rng.normal(0.0, 1e-4, n)
    pz = 5e5 + 3e5 * (x / 1e-4)
    px = 1e4 * (x / 1e-4)
    py = 1e4 * (y / 1e-4)
    return ParticleDistribution3D(
        x=x, y=y, z=np.zeros(n),
        px=px, py=py, pz=pz,
        t=np.zeros(n), Q=np.full(n, -1.6e-19),
    )


def _make_wide_spread_slice(n=5000, seed=5):
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1e-4, n)
    y = rng.normal(0.0, 1e-4, n)
    pz = 5e5 + 3e5 * (x / 1e-4)
    px = 1e4 * (x / 1e-4)
    py = 1e4 * (y / 1e-4)
    return SliceDistribution(
        z=0.0,
        x=x, y=y,
        px=px, py=py, pz=pz,
        t=np.zeros(n), lam=np.full(n, 1e-9),
    )


def test_pd3d_nemit_x_matches_compute_phase_space_plane():
    d = _make_wide_spread_3d()
    expected = compute_phase_space_plane(d, plane="x", weight="Q_abs").normalized_emittance
    np.testing.assert_allclose(d.nemit_x, expected, rtol=1e-10)


def test_pd3d_nemit_y_matches_compute_phase_space_plane():
    d = _make_wide_spread_3d()
    expected = compute_phase_space_plane(d, plane="y", weight="Q_abs").normalized_emittance
    np.testing.assert_allclose(d.nemit_y, expected, rtol=1e-10)


def test_pdslice_nemit_x_matches_compute_phase_space_plane():
    s = _make_wide_spread_slice()
    expected = compute_phase_space_plane(s, plane="x", weight="lam_abs").normalized_emittance
    np.testing.assert_allclose(s.nemit_x, expected, rtol=1e-10)


def test_pdslice_nemit_y_matches_compute_phase_space_plane():
    s = _make_wide_spread_slice()
    expected = compute_phase_space_plane(s, plane="y", weight="lam_abs").normalized_emittance
    np.testing.assert_allclose(s.nemit_y, expected, rtol=1e-10)


def test_pd3d_nemit_x_differs_from_old_formula_for_wide_spread():
    """Sanity check: the mildly-relativistic fixture exercises the new formula's divergence
    from the old `beta0 * gamma0` (single-particle from <E_k>) formula at ≥1% level."""
    d = _make_wide_spread_3d()
    old = d.beta0 * d.gamma0 * d.emit_x
    new = d.nemit_x
    rel = abs(new - old) / max(abs(old), 1e-300)
    assert rel > 1e-3, f"Fixture not wide enough: rel diff={rel}"
