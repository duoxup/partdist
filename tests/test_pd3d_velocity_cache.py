"""Performance regression test: _calc_velocities is called once per access chain
and invalidates correctly on base-quantity updates."""
import numpy as np
from unittest.mock import patch

from partdist import ParticleDistribution3D


def _make_dist(n=10_000):
    rng = np.random.default_rng(3)
    return ParticleDistribution3D(
        x=np.zeros(n), y=np.zeros(n), z=np.zeros(n),
        px=rng.normal(0, 1e5, n), py=rng.normal(0, 1e5, n),
        pz=1e7 + rng.normal(0, 1e5, n),
        t=np.zeros(n), Q=np.full(n, -1.6e-19),
    )


def test_velocities_computed_once_for_vx_vy_vz_chain():
    """Accessing vx then vy then vz must trigger exactly one underlying conversion."""
    d = _make_dist()
    from partdist.pd3d import utils as pd3d_utils

    call_count = 0
    original = pd3d_utils.momentum_evc_to_velocity

    def counting_call(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original(*args, **kwargs)

    with patch.object(pd3d_utils, "momentum_evc_to_velocity", counting_call):
        _ = d.vx
        _ = d.vy
        _ = d.vz

    assert call_count == 1, f"Expected 1 conversion call, got {call_count}"


def test_cache_invalidates_on_px_update():
    """After update_quantity('px', ...), vx should reflect the new value."""
    d = _make_dist()
    vx_before = d.vx.copy()
    new_px = d.get_data("px") + 1e6
    d.update_quantity("px", new_px)
    vx_after = d.vx
    assert not np.allclose(vx_before, vx_after), "vx must change after px update"


def test_cache_invalidates_on_pz_update():
    d = _make_dist()
    vx_before = d.vx.copy()
    new_pz = d.get_data("pz") * 2.0
    d.update_quantity("pz", new_pz)
    vx_after = d.vx
    assert not np.allclose(vx_before, vx_after), "vx must change after pz update"
