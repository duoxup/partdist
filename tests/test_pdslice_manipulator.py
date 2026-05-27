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


from partdist.pdslice.manipulator import shift_centroid


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
