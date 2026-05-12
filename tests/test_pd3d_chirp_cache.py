"""Performance regression test: chirp polynomial coefficients are computed once
per quadratic/cubic_chirp access chain and invalidate correctly on base updates."""
import numpy as np

from partdist import ParticleDistribution3D


def _make_chirped_dist(n=5000):
    rng = np.random.default_rng(4)
    z = rng.normal(0, 1e-3, n)
    return ParticleDistribution3D(
        x=np.zeros(n), y=np.zeros(n), z=z,
        px=np.zeros(n), py=np.zeros(n),
        pz=1e7 + 1e6 * (z / 1e-3) + rng.normal(0, 1e4, n),
        t=np.zeros(n), Q=np.full(n, -1.6e-19),
    )


def test_chirp_polyfit_runs_once_for_quadratic_and_cubic():
    """Accessing quadratic_chirp then cubic_chirp should trigger exactly one polyfit."""
    d = _make_chirped_dist()
    real_calc = d._calc_chirp_poly_coeffs
    count = 0

    def counting():
        nonlocal count
        count += 1
        return real_calc()

    d._calc_chirp_poly_coeffs = counting
    _ = d.quadratic_chirp
    _ = d.cubic_chirp
    assert count == 1, f"Expected 1 polyfit call, got {count}"


def test_chirp_cache_invalidates_on_pz_update():
    d = _make_chirped_dist()
    q_before = d.quadratic_chirp
    new_pz = d.get_data("pz") * 1.5    # change pz; chirp coefficient must change
    d.update_quantity("pz", new_pz)
    q_after = d.quadratic_chirp
    assert q_before != q_after, "quadratic_chirp must change after pz update"


def test_chirp_cache_invalidates_on_z_update():
    d = _make_chirped_dist()
    q_before = d.quadratic_chirp
    new_z = d.get_data("z") * 2.0
    d.update_quantity("z", new_z)
    q_after = d.quadratic_chirp
    assert q_before != q_after, "quadratic_chirp must change after z update"
