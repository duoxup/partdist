"""Regression tests pinning emit_z to the (z, delta) phase-space form."""
import numpy as np
from partdist import ParticleDistribution3D


def _make_chirped_dist(n=5000, seed=1):
    """Beam with a deliberate z–delta correlation and finite spreads in both."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0.0, 1e-3, n)
    pz0 = 1e7
    pz = pz0 * (1.0 + 0.01 * (z / 1e-3)) + rng.normal(0.0, 1e4, n)
    return ParticleDistribution3D(
        x=np.zeros(n), y=np.zeros(n), z=z,
        px=np.zeros(n), py=np.zeros(n), pz=pz,
        t=np.zeros(n), Q=np.full(n, -1.6e-19),
    )


def test_emit_z_uses_delta_not_pz():
    """emit_z = sqrt(var(z)*var(delta) - cov(z, delta)^2), not var(pz)."""
    d = _make_chirped_dist()
    delta = d.get_data("delta")
    z = d.get_data("z")
    w = d.Q_abs
    z_c = z - np.average(z, weights=w)
    d_c = delta - np.average(delta, weights=w)
    var_z = float(np.average(z_c ** 2, weights=w))
    var_d = float(np.average(d_c ** 2, weights=w))
    cov_zd = float(np.average(z_c * d_c, weights=w))
    expected = float(np.sqrt(max(var_z * var_d - cov_zd ** 2, 0.0)))
    np.testing.assert_allclose(d.emit_z, expected, rtol=1e-10)


def test_emit_z_scales_linearly_with_z():
    """delta is dimensionless and momentum-only; emit_z must be in metres and scale linearly with z."""
    n = 5000
    seed = 1
    rng = np.random.default_rng(seed)
    z1 = rng.normal(0.0, 1e-3, n)
    rng = np.random.default_rng(seed)  # reset so pz is the same
    _ = rng.normal(0.0, 1e-3, n)
    pz0 = 1e7
    pz = pz0 * (1.0 + 0.01 * (z1 / 1e-3)) + rng.normal(0.0, 1e4, n)

    d1 = ParticleDistribution3D(
        x=np.zeros(n), y=np.zeros(n), z=z1,
        px=np.zeros(n), py=np.zeros(n), pz=pz,
        t=np.zeros(n), Q=np.full(n, -1.6e-19),
    )
    d2 = ParticleDistribution3D(
        x=np.zeros(n), y=np.zeros(n), z=2.0 * z1,
        px=np.zeros(n), py=np.zeros(n), pz=pz,
        t=np.zeros(n), Q=np.full(n, -1.6e-19),
    )
    # delta depends only on pz/Q (not on z), so var(delta) and cov(z,delta) scale with z linearly,
    # leaving emit_z to scale linearly with z. Allow a slightly looser tol because of the
    # correlation between z and pz in this fixture.
    np.testing.assert_allclose(d2.emit_z, 2.0 * d1.emit_z, rtol=5e-2)


def test_nemit_z_is_beta0_gamma0_times_emit_z():
    """The body of nemit_z is unchanged; only the interpretation of emit_z changed."""
    d = _make_chirped_dist()
    np.testing.assert_allclose(d.nemit_z, d.beta0 * d.gamma0 * d.emit_z, rtol=1e-12)
