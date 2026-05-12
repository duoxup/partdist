"""Regression tests for momentum_evc(copy=...) semantics on both containers."""
import numpy as np

from partdist import ParticleDistribution3D, SliceDistribution


def _make_3d(n=100):
    rng = np.random.default_rng(0)
    return ParticleDistribution3D.from_arrays(
        x=rng.normal(0, 1e-4, n),
        y=rng.normal(0, 1e-4, n),
        z=rng.normal(0, 1e-3, n),
        px=rng.normal(0, 1e5, n),
        py=rng.normal(0, 1e5, n),
        pz=1e7 + rng.normal(0, 1e5, n),
        t=np.zeros(n),
        Q=np.full(n, -1.6e-19),
    )


def _make_slice(n=100):
    rng = np.random.default_rng(1)
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


def test_pd3d_default_copy_protects_internal_state():
    d = _make_3d()
    px_internal_before = d.px.copy()
    px, _, _ = d.momentum_evc()
    px[:] = 0.0
    np.testing.assert_array_equal(d.px, px_internal_before)


def test_pd3d_copy_false_aliases_internal_data():
    d = _make_3d()
    px, py, pz = d.momentum_evc(copy=False)
    assert px is d._quantities["px"].data
    assert py is d._quantities["py"].data
    assert pz is d._quantities["pz"].data


def test_pd3d_copy_true_and_false_numerically_equivalent():
    d = _make_3d()
    px_c, py_c, pz_c = d.momentum_evc(copy=True)
    px_nc, py_nc, pz_nc = d.momentum_evc(copy=False)
    np.testing.assert_array_equal(px_c, px_nc)
    np.testing.assert_array_equal(py_c, py_nc)
    np.testing.assert_array_equal(pz_c, pz_nc)


def test_slice_default_copy_protects_internal_state():
    s = _make_slice()
    px_internal_before = s.px.copy()
    px, _, _ = s.momentum_evc()
    px[:] = 0.0
    np.testing.assert_array_equal(s.px, px_internal_before)


def test_slice_copy_false_aliases_internal_data():
    s = _make_slice()
    px, py, pz = s.momentum_evc(copy=False)
    assert px is s._quantities["px"].data
    assert py is s._quantities["py"].data
    assert pz is s._quantities["pz"].data


def test_slice_copy_true_and_false_numerically_equivalent():
    s = _make_slice()
    px_c, py_c, pz_c = s.momentum_evc(copy=True)
    px_nc, py_nc, pz_nc = s.momentum_evc(copy=False)
    np.testing.assert_array_equal(px_c, px_nc)
    np.testing.assert_array_equal(py_c, py_nc)
    np.testing.assert_array_equal(pz_c, pz_nc)
