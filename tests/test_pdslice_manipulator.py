"""Tests for partdist.pdslice.manipulator."""
from __future__ import annotations

import math
import numpy as np
import pytest

from partdist.pdslice.generator import (
    Gaussian, Uniform, Plateau, RadialUniform, Isotropic, make_slice,
)
from partdist.pdslice import SliceDistribution


def gauss_slice(n: int = 5000, seed: int = 0) -> SliceDistribution:
    """Standard test fixture: 5D Gaussian slice."""
    return make_slice(
        n,
        I_total=1e-3,
        x=Gaussian(sig=2e-4), y=Gaussian(sig=3e-4),
        px=Gaussian(sig=1e3), py=Gaussian(sig=2e3),
        pz=Gaussian(sig=1e3), pz_anchor=5e5,
        seed=seed,
    )


class TestModuleImports:
    def test_module_imports(self):
        from partdist.pdslice import manipulator
        assert manipulator is not None


from partdist.pdslice.manipulator import shift_centroid, rotate_xy


class TestShiftCentroid:
    def test_dx_shifts_x_mean(self):
        d = gauss_slice()
        x_before = d.get_data("x").mean()
        out = shift_centroid(d, dx=1e-3)
        x_after = out.get_data("x").mean()
        assert abs((x_after - x_before) - 1e-3) < 1e-15

    def test_other_axes_unchanged_when_only_dx_set(self):
        d = gauss_slice()
        y_before = d.get_data("y").copy()
        px_before = d.get_data("px").copy()
        out = shift_centroid(d, dx=1e-3)
        np.testing.assert_array_equal(out.get_data("y"), y_before)
        np.testing.assert_array_equal(out.get_data("px"), px_before)

    def test_all_axes_simultaneously(self):
        d = gauss_slice()
        out = shift_centroid(d, dx=1e-3, dy=2e-3, dpx=10.0, dpy=20.0, dpz=30.0)
        assert abs(out.get_data("x").mean() - d.get_data("x").mean() - 1e-3) < 1e-15
        assert abs(out.get_data("y").mean() - d.get_data("y").mean() - 2e-3) < 1e-15
        assert abs(out.get_data("px").mean() - d.get_data("px").mean() - 10.0) < 1e-10
        assert abs(out.get_data("py").mean() - d.get_data("py").mean() - 20.0) < 1e-10
        assert abs(out.get_data("pz").mean() - d.get_data("pz").mean() - 30.0) < 1e-10

    def test_mask_isolates_changes(self):
        d = gauss_slice()
        x_before = d.get_data("x").copy()
        mask = np.zeros(len(d), dtype=bool)
        mask[:100] = True
        out = shift_centroid(d, dx=1.0, mask=mask)
        x_after = out.get_data("x")
        np.testing.assert_array_almost_equal(x_after[:100], x_before[:100] + 1.0)
        np.testing.assert_array_equal(x_after[100:], x_before[100:])


class TestRotateXY:
    def test_zero_angle_is_identity(self):
        d = gauss_slice()
        out = rotate_xy(d, theta=0.0)
        np.testing.assert_array_equal(out.get_data("x"), d.get_data("x"))
        np.testing.assert_array_equal(out.get_data("y"), d.get_data("y"))

    def test_two_pi_is_identity_to_fp_tolerance(self):
        d = gauss_slice()
        out = rotate_xy(d, theta=2.0 * math.pi)
        np.testing.assert_allclose(out.get_data("x"), d.get_data("x"), atol=1e-14)
        np.testing.assert_allclose(out.get_data("y"), d.get_data("y"), atol=1e-14)

    def test_pi_over_two_swaps_sigmas(self):
        d = gauss_slice()
        sx_before = d.get_data("x").std()
        sy_before = d.get_data("y").std()
        out = rotate_xy(d, theta=math.pi / 2.0)
        assert abs(out.get_data("x").std() - sy_before) < 1e-12
        assert abs(out.get_data("y").std() - sx_before) < 1e-12

    def test_radius_invariant(self):
        d = gauss_slice()
        r_sq_before = d.get_data("x") ** 2 + d.get_data("y") ** 2
        out = rotate_xy(d, theta=0.3)
        r_sq_after = out.get_data("x") ** 2 + out.get_data("y") ** 2
        np.testing.assert_allclose(r_sq_after, r_sq_before, rtol=1e-14)

    def test_momentum_rotates_consistently(self):
        d = gauss_slice()
        p_sq_before = d.get_data("px") ** 2 + d.get_data("py") ** 2
        out = rotate_xy(d, theta=0.5)
        p_sq_after = out.get_data("px") ** 2 + out.get_data("py") ** 2
        np.testing.assert_allclose(p_sq_after, p_sq_before, rtol=1e-14)


from partdist.pdslice.manipulator import center_beam


def _weighted_mean(arr, w):
    return float(np.sum(arr * w) / np.sum(w))


class TestCenterBeam:
    def test_centers_default_axes(self):
        d = shift_centroid(gauss_slice(), dx=5e-4, dy=-3e-4, dpx=50.0, dpy=-30.0)
        out = center_beam(d)
        w = np.abs(out.get_data("lam"))
        for k in ("x", "y", "px", "py"):
            assert abs(_weighted_mean(out.get_data(k), w)) < 1e-10

    def test_pz_unchanged_by_default(self):
        d = gauss_slice()
        pz_before = d.get_data("pz").copy()
        out = center_beam(d)
        np.testing.assert_array_equal(out.get_data("pz"), pz_before)

    def test_explicit_pz_axis_centers_pz(self):
        d = gauss_slice()
        out = center_beam(d, axes=("x", "y", "px", "py", "pz"))
        w = np.abs(out.get_data("lam"))
        assert abs(_weighted_mean(out.get_data("pz"), w)) < 1e-5  # pz scale ~ 5e5

    def test_subset_axes(self):
        d = shift_centroid(gauss_slice(), dx=5e-4, dy=-3e-4)
        y_before = d.get_data("y").copy()
        out = center_beam(d, axes=("x",))
        w = np.abs(out.get_data("lam"))
        assert abs(_weighted_mean(out.get_data("x"), w)) < 1e-10
        np.testing.assert_array_equal(out.get_data("y"), y_before)
