"""Performance regression test: SliceDistribution._calc_velocities is computed once
per access chain and invalidates correctly on base-quantity updates.

Mirror of tests/test_pd3d_velocity_cache.py for the slice container.
"""
import numpy as np
from unittest.mock import patch

from partdist import SliceDistribution


def _make_slice(n=10_000):
    rng = np.random.default_rng(3)
    return SliceDistribution(
        z=0.0,
        x=rng.normal(0, 1e-4, n),
        y=rng.normal(0, 1e-4, n),
        px=rng.normal(0, 1e5, n),
        py=rng.normal(0, 1e5, n),
        pz=1e7 + rng.normal(0, 1e5, n),
        t=np.zeros(n),
        lam=np.full(n, 1e-9),
    )


def test_velocities_computed_once_for_vx_vy_vz_chain():
    """Accessing vx then vy then vz must trigger exactly one underlying conversion."""
    s = _make_slice()
    from partdist.pd3d import utils as pd3d_utils

    call_count = 0
    original = pd3d_utils.momentum_evc_to_velocity

    def counting_call(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original(*args, **kwargs)

    with patch.object(pd3d_utils, "momentum_evc_to_velocity", counting_call):
        _ = s.vx
        _ = s.vy
        _ = s.vz

    assert call_count == 1, f"Expected 1 conversion call, got {call_count}"


def test_cache_invalidates_on_px_update():
    """After update_quantity('px', ...), vx should reflect the new value."""
    s = _make_slice()
    vx_before = s.vx.copy()
    new_px = s.get_data("px") + 1e6
    s.update_quantity("px", new_px)
    vx_after = s.vx
    assert not np.allclose(vx_before, vx_after), "vx must change after px update"


def test_cache_invalidates_on_pz_update():
    s = _make_slice()
    vx_before = s.vx.copy()
    new_pz = s.get_data("pz") * 2.0
    s.update_quantity("pz", new_pz)
    vx_after = s.vx
    assert not np.allclose(vx_before, vx_after), "vx must change after pz update"
