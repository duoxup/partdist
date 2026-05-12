"""Regression test: normalised emittance uses beta0*gamma0 of the reference particle,
not the per-particle weighted mean of beta*gamma.
"""
import numpy as np
from partdist import ParticleDistribution3D
from partdist.pd3d.analysis import compute_phase_space_plane


def _make_energy_correlated_dist(n=4000, seed=2):
    """Beam with strong x-correlated pz spread, so <beta*gamma> != beta0*gamma0."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1e-4, n)
    pz = 1e8 + 5e7 * (x / 1e-4)   # 50% spread, x-correlated
    px = 1e5 * (x / 1e-4)         # x-correlated kick
    return ParticleDistribution3D(
        x=x, y=np.zeros(n), z=np.zeros(n),
        px=px, py=np.zeros(n), pz=pz,
        t=np.zeros(n), Q=np.full(n, -1.6e-19),
    )


def test_nemit_x_uses_reference_beta_gamma():
    """eps_norm == beta0*gamma0 * eps_geom (reference-particle convention)."""
    d = _make_energy_correlated_dist()
    res = compute_phase_space_plane(d, plane="x", weight="Q_abs")

    # Confirmed field names from PhaseSpacePlaneResult dataclass:
    #   .geometric_emittance  and  .normalized_emittance
    eps_geom = res.geometric_emittance
    eps_norm = res.normalized_emittance

    # Independently compute the expected beta0*gamma0 from the charge-weighted reference
    # momentum, using the same selection that the analysis function uses (all-finite,
    # |pz| > 0, w > 0). For this fixture all particles are valid so we can compute
    # directly from d.p_abs_si.
    p_si = d.get_data("p_abs_si")
    w = d.Q_abs
    p_ref = float(np.average(p_si, weights=w))

    from scipy.constants import c as c_light
    from scipy.constants import m_e
    m0 = m_e
    beta_gamma_ref = p_ref / (m0 * c_light)
    expected = beta_gamma_ref * eps_geom

    # Old formula <p/(m0 c)> would give the WEIGHTED MEAN of beta*gamma, which
    # for this energy-correlated fixture differs from beta0*gamma0 by a measurable amount.
    np.testing.assert_allclose(eps_norm, expected, rtol=1e-10)
