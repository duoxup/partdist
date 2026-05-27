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


from partdist.pdslice.manipulator import match_twiss_x
from partdist.pd3d.analysis import compute_phase_space_plane


class TestMatchTwissX:
    def test_recovers_target_alpha_beta(self):
        d = gauss_slice()
        out = match_twiss_x(d, alpha=2.0, beta=5.0)
        result = compute_phase_space_plane(out, plane="x", weight="lam_abs")
        assert abs(result.alpha - 2.0) < 1e-10
        assert abs(result.beta - 5.0) < 1e-10

    def test_emittance_preserved(self):
        d = gauss_slice()
        eps_before = compute_phase_space_plane(d, plane="x", weight="lam_abs").geometric_emittance
        out = match_twiss_x(d, alpha=1.5, beta=3.0)
        eps_after = compute_phase_space_plane(out, plane="x", weight="lam_abs").geometric_emittance
        assert abs(eps_after - eps_before) / eps_before < 1e-12

    def test_y_plane_unaffected(self):
        d = gauss_slice()
        y_before = d.get_data("y").copy()
        py_before = d.get_data("py").copy()
        out = match_twiss_x(d, alpha=2.0, beta=5.0)
        np.testing.assert_array_equal(out.get_data("y"), y_before)
        np.testing.assert_array_equal(out.get_data("py"), py_before)

    def test_preserve_centroid(self):
        d = shift_centroid(gauss_slice(), dx=1e-3, dpx=100.0)
        x_mean_before = d.get_data("x").mean()
        out = match_twiss_x(d, alpha=1.0, beta=2.0, preserve_centroid=True)
        # weighted mean stays approximately at original
        w = np.abs(out.get_data("lam"))
        x_mean_after = _weighted_mean(out.get_data("x"), w)
        assert abs(x_mean_after - x_mean_before) < 1e-10

    def test_rejects_nonpositive_beta(self):
        d = gauss_slice()
        with pytest.raises(ValueError, match="beta"):
            match_twiss_x(d, alpha=1.0, beta=0.0)
        with pytest.raises(ValueError, match="beta"):
            match_twiss_x(d, alpha=1.0, beta=-1.0)


from partdist.pdslice.manipulator import match_twiss_y


class TestMatchTwissY:
    def test_recovers_target_alpha_beta(self):
        d = gauss_slice()
        out = match_twiss_y(d, alpha=1.0, beta=4.0)
        result = compute_phase_space_plane(out, plane="y", weight="lam_abs")
        assert abs(result.alpha - 1.0) < 1e-10
        assert abs(result.beta - 4.0) < 1e-10

    def test_x_plane_unaffected(self):
        d = gauss_slice()
        x_before = d.get_data("x").copy()
        px_before = d.get_data("px").copy()
        out = match_twiss_y(d, alpha=1.0, beta=4.0)
        np.testing.assert_array_equal(out.get_data("x"), x_before)
        np.testing.assert_array_equal(out.get_data("px"), px_before)


from partdist.pdslice.manipulator import match_twiss_xy


class TestMatchTwissXY:
    def test_recovers_both_targets(self):
        d = gauss_slice()
        out = match_twiss_xy(d, alpha_x=2.0, beta_x=5.0, alpha_y=1.0, beta_y=4.0)
        rx = compute_phase_space_plane(out, plane="x", weight="lam_abs")
        ry = compute_phase_space_plane(out, plane="y", weight="lam_abs")
        assert abs(rx.alpha - 2.0) < 1e-10 and abs(rx.beta - 5.0) < 1e-10
        assert abs(ry.alpha - 1.0) < 1e-10 and abs(ry.beta - 4.0) < 1e-10

    def test_order_independent(self):
        d = gauss_slice()
        out_xy = match_twiss_xy(d, alpha_x=2.0, beta_x=5.0, alpha_y=1.0, beta_y=4.0)
        # equivalent: apply y first, then x
        out_yx = match_twiss_x(
            match_twiss_y(d, alpha=1.0, beta=4.0),
            alpha=2.0, beta=5.0,
        )
        np.testing.assert_allclose(out_xy.get_data("x"), out_yx.get_data("x"), atol=1e-12)
        np.testing.assert_allclose(out_xy.get_data("y"), out_yx.get_data("y"), atol=1e-12)


from partdist.pdslice.manipulator import apply_dispersion


class TestApplyDispersion:
    def test_recovers_D_from_linear_regression_x(self):
        d = gauss_slice(n=20_000)
        D = 5e-2
        out = apply_dispersion(d, D=D, axis="x")
        x = out.get_data("x")
        pz = out.get_data("pz")
        w = np.abs(out.get_data("lam"))
        wsum = float(np.sum(w))
        p_ref = float(np.sum(pz * w) / wsum)
        delta = (pz - p_ref) / p_ref
        # weighted slope of x vs delta around the new centroid
        mean_x = float(np.sum(x * w) / wsum)
        mean_d = float(np.sum(delta * w) / wsum)
        num = float(np.sum(w * (x - mean_x) * (delta - mean_d)))
        den = float(np.sum(w * (delta - mean_d) ** 2))
        slope = num / den
        assert abs(slope - D) / D < 0.05

    def test_axis_y_independent_of_x(self):
        d = gauss_slice()
        x_before = d.get_data("x").copy()
        out = apply_dispersion(d, D=1e-2, axis="y")
        np.testing.assert_array_equal(out.get_data("x"), x_before)

    def test_explicit_p_ref(self):
        d = gauss_slice()
        out = apply_dispersion(d, D=1e-2, axis="x", p_ref=5e5)
        delta = (d.get_data("pz") - 5e5) / 5e5
        expected = d.get_data("x") + 1e-2 * delta
        np.testing.assert_allclose(out.get_data("x"), expected, rtol=1e-12)

    def test_rejects_bad_axis(self):
        d = gauss_slice()
        with pytest.raises(ValueError, match="axis"):
            apply_dispersion(d, D=1e-2, axis="z")

    def test_rejects_zero_p_ref(self):
        d = gauss_slice()
        with pytest.raises(ValueError, match="p_ref"):
            apply_dispersion(d, D=1e-2, p_ref=0.0)


from partdist.pdslice.manipulator import scale_rms_x


class TestScaleRMSX:
    def test_factor_doubles_sigma_x(self):
        d = gauss_slice(n=20_000)
        sx_before = d.get_data("x").std()
        out = scale_rms_x(d, factor=2.0)
        assert abs(out.get_data("x").std() - 2.0 * sx_before) < 1e-10

    def test_emittance_preserving_keeps_emittance(self):
        d = gauss_slice(n=20_000)
        eps_before = compute_phase_space_plane(d, plane="x", weight="lam_abs").geometric_emittance
        out = scale_rms_x(d, factor=2.0, emittance_preserving=True)
        eps_after = compute_phase_space_plane(out, plane="x", weight="lam_abs").geometric_emittance
        assert abs(eps_after - eps_before) / eps_before < 1e-12

    def test_not_emittance_preserving_scales_emittance(self):
        d = gauss_slice(n=20_000)
        eps_before = compute_phase_space_plane(d, plane="x", weight="lam_abs").geometric_emittance
        out = scale_rms_x(d, factor=2.0, emittance_preserving=False)
        eps_after = compute_phase_space_plane(out, plane="x", weight="lam_abs").geometric_emittance
        assert abs(eps_after - 2.0 * eps_before) / eps_before < 1e-12

    def test_y_plane_unaffected(self):
        d = gauss_slice()
        y_before = d.get_data("y").copy()
        py_before = d.get_data("py").copy()
        out = scale_rms_x(d, factor=1.5)
        np.testing.assert_array_equal(out.get_data("y"), y_before)
        np.testing.assert_array_equal(out.get_data("py"), py_before)

    def test_rejects_nonpositive_factor(self):
        d = gauss_slice()
        with pytest.raises(ValueError, match="factor"):
            scale_rms_x(d, factor=0.0)
        with pytest.raises(ValueError, match="factor"):
            scale_rms_x(d, factor=-1.0)


from partdist.pdslice.manipulator import scale_rms_y


class TestScaleRMSY:
    def test_factor_doubles_sigma_y(self):
        d = gauss_slice(n=20_000)
        sy_before = d.get_data("y").std()
        out = scale_rms_y(d, factor=2.0)
        assert abs(out.get_data("y").std() - 2.0 * sy_before) < 1e-10

    def test_x_plane_unaffected(self):
        d = gauss_slice()
        x_before = d.get_data("x").copy()
        px_before = d.get_data("px").copy()
        out = scale_rms_y(d, factor=1.5)
        np.testing.assert_array_equal(out.get_data("x"), x_before)
        np.testing.assert_array_equal(out.get_data("px"), px_before)

    def test_rejects_nonpositive_factor(self):
        d = gauss_slice()
        with pytest.raises(ValueError, match="factor"):
            scale_rms_y(d, factor=0.0)
        with pytest.raises(ValueError, match="factor"):
            scale_rms_y(d, factor=-1.0)


class TestSharedBehaviour:
    @pytest.mark.parametrize("op", [
        lambda d, **kw: match_twiss_x(d, alpha=1.0, beta=3.0, **kw),
        lambda d, **kw: apply_dispersion(d, D=1e-2, **kw),
        lambda d, **kw: center_beam(d, **kw),
    ])
    def test_inplace_false_returns_new_instance(self, op):
        d = gauss_slice()
        x_before = d.get_data("x").copy()
        out = op(d)
        assert out is not d
        np.testing.assert_array_equal(d.get_data("x"), x_before)

    @pytest.mark.parametrize("op", [
        lambda d, **kw: match_twiss_x(d, alpha=1.0, beta=3.0, **kw),
        lambda d, **kw: apply_dispersion(d, D=1e-2, **kw),
        lambda d, **kw: center_beam(d, **kw),
    ])
    def test_inplace_true_returns_same_instance(self, op):
        d = gauss_slice()
        out = op(d, inplace=True)
        assert out is d

    @pytest.mark.parametrize("op", [
        lambda d, **kw: match_twiss_x(d, alpha=1.0, beta=3.0, **kw),
        lambda d, **kw: apply_dispersion(d, D=1e-2, **kw),
        lambda d, **kw: center_beam(d, **kw),
    ])
    def test_mask_leaves_unmasked_particles_unchanged(self, op):
        d = gauss_slice()
        x_before = d.get_data("x").copy()
        y_before = d.get_data("y").copy()
        px_before = d.get_data("px").copy()
        py_before = d.get_data("py").copy()
        mask = np.zeros(len(d), dtype=bool)
        mask[:100] = True
        out = op(d, mask=mask)
        np.testing.assert_array_equal(out.get_data("x")[100:], x_before[100:])
        np.testing.assert_array_equal(out.get_data("y")[100:], y_before[100:])
        np.testing.assert_array_equal(out.get_data("px")[100:], px_before[100:])
        np.testing.assert_array_equal(out.get_data("py")[100:], py_before[100:])

    @pytest.mark.parametrize("op", [
        lambda d, **kw: match_twiss_x(d, alpha=1.0, beta=3.0, **kw),
        lambda d, **kw: apply_dispersion(d, D=1e-2, **kw),
        lambda d, **kw: center_beam(d, **kw),
    ])
    def test_weight_accepts_three_forms_equivalently(self, op):
        """For a uniform-lam beam, weight=None, weight="lam_abs", and an
        explicit uniform array should all give identical output."""
        d = gauss_slice()
        out_none = op(d, weight=None)
        out_str = op(d, weight="lam_abs")
        out_arr = op(d, weight=np.full(len(d), 1.0))
        for key in ("x", "y", "px", "py"):
            np.testing.assert_allclose(
                out_none.get_data(key), out_str.get_data(key), rtol=1e-12, atol=1e-15,
            )
            np.testing.assert_allclose(
                out_str.get_data(key), out_arr.get_data(key), rtol=1e-12, atol=1e-15,
            )
