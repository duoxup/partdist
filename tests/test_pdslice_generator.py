"""Tests for partdist.pdslice.generator."""
from __future__ import annotations

import math
import numpy as np
import pytest

from partdist.pdslice.generator import (
    Gaussian, Uniform, Plateau, RadialUniform, Isotropic, make_slice,
)


class TestShapeConstruction:
    def test_gaussian_basic(self):
        g = Gaussian(sig=1e-4)
        assert g.sig == 1e-4
        assert g.mean == 0.0
        assert g.cut is None

    def test_gaussian_with_cut(self):
        g = Gaussian(sig=1e-4, mean=5e5, cut=3.0)
        assert g.cut == 3.0

    def test_gaussian_rejects_nonpositive_sig(self):
        with pytest.raises(ValueError, match="sig"):
            Gaussian(sig=0.0)
        with pytest.raises(ValueError, match="sig"):
            Gaussian(sig=-1.0)

    def test_gaussian_rejects_nonpositive_cut(self):
        with pytest.raises(ValueError, match="cut"):
            Gaussian(sig=1.0, cut=0.0)
        with pytest.raises(ValueError, match="cut"):
            Gaussian(sig=1.0, cut=-1.0)

    def test_uniform_basic(self):
        u = Uniform(L=1e-3)
        assert u.L == 1e-3
        assert u.mean == 0.0

    def test_uniform_rejects_nonpositive_L(self):
        with pytest.raises(ValueError, match="L"):
            Uniform(L=0.0)
        with pytest.raises(ValueError, match="L"):
            Uniform(L=-1e-3)

    def test_plateau_basic(self):
        p = Plateau(L=1e-3, r=1e-4)
        assert p.L == 1e-3
        assert p.r == 1e-4

    def test_plateau_rejects_nonpositive_params(self):
        with pytest.raises(ValueError, match="L"):
            Plateau(L=0.0, r=1e-4)
        with pytest.raises(ValueError, match="r"):
            Plateau(L=1e-3, r=0.0)

    def test_radial_uniform_basic(self):
        r = RadialUniform(R=1e-3)
        assert r.R == 1e-3

    def test_radial_uniform_rejects_nonpositive_R(self):
        with pytest.raises(ValueError, match="R"):
            RadialUniform(R=0.0)

    def test_isotropic_basic(self):
        iso = Isotropic(p_mag=1e5)
        assert iso.p_mag == 1e5

    def test_isotropic_rejects_nonpositive_p_mag(self):
        with pytest.raises(ValueError, match="p_mag"):
            Isotropic(p_mag=0.0)

    def test_shapes_are_frozen(self):
        g = Gaussian(sig=1e-4)
        with pytest.raises(Exception):
            g.sig = 2e-4


class TestGaussianSampling:
    def test_untruncated_size_and_stats(self):
        g = Gaussian(sig=2.0, mean=5.0)
        rng = np.random.default_rng(0)
        samples = g._sample(50_000, rng)
        assert samples.shape == (50_000,)
        assert abs(samples.mean() - 5.0) < 0.05
        assert abs(samples.std() - 2.0) < 0.05

    def test_truncated_respects_bounds(self):
        g = Gaussian(sig=1.0, mean=0.0, cut=2.0)
        rng = np.random.default_rng(1)
        samples = g._sample(10_000, rng)
        assert samples.shape == (10_000,)
        assert np.all(np.abs(samples) <= 2.0 + 1e-12), \
            "Truncated samples must lie within ±cut·sig of mean"

    def test_truncated_rms_smaller_than_input_sig(self):
        """At cut=2σ, the truncated distribution's rms < input σ."""
        g = Gaussian(sig=1.0, mean=0.0, cut=2.0)
        rng = np.random.default_rng(2)
        samples = g._sample(50_000, rng)
        actual_rms = samples.std()
        assert 0.85 < actual_rms < 0.90, \
            f"Expected truncated rms ≈ 0.879, got {actual_rms}"

    def test_truncated_with_offset_mean(self):
        g = Gaussian(sig=1.0, mean=10.0, cut=3.0)
        rng = np.random.default_rng(3)
        samples = g._sample(10_000, rng)
        assert np.all((samples >= 7.0) & (samples <= 13.0))
        assert abs(samples.mean() - 10.0) < 0.05

    def test_reproducible_with_same_rng(self):
        g = Gaussian(sig=1.0)
        rng_a = np.random.default_rng(42)
        rng_b = np.random.default_rng(42)
        a = g._sample(1000, rng_a)
        b = g._sample(1000, rng_b)
        np.testing.assert_array_equal(a, b)


class TestUniformSampling:
    def test_bounds(self):
        u = Uniform(L=2.0, mean=10.0)
        rng = np.random.default_rng(0)
        samples = u._sample(10_000, rng)
        assert samples.shape == (10_000,)
        assert np.all((samples >= 9.0) & (samples <= 11.0))

    def test_stats_match_uniform(self):
        u = Uniform(L=1.0, mean=0.0)
        rng = np.random.default_rng(1)
        samples = u._sample(50_000, rng)
        assert abs(samples.mean()) < 0.01
        expected_sigma = 1.0 / (2.0 * math.sqrt(3.0))
        assert abs(samples.std() - expected_sigma) < 0.005


class TestPlateauSampling:
    def test_bounded_envelope(self):
        """Samples must fall inside the rejection-sampling window."""
        p = Plateau(L=1.0, r=0.1, mean=0.0)
        rng = np.random.default_rng(0)
        samples = p._sample(20_000, rng)
        assert samples.shape == (20_000,)
        bound = 0.5 + 5.0 * 0.1
        assert np.all(np.abs(samples) <= bound + 1e-12)

    def test_symmetric_about_mean(self):
        p = Plateau(L=2.0, r=0.2, mean=3.0)
        rng = np.random.default_rng(1)
        samples = p._sample(50_000, rng)
        assert abs(samples.mean() - 3.0) < 0.02

    def test_sigma_in_expected_range(self):
        """ASTRA: L/(2√3) ≤ σ ≤ L/2.8 for the plateau family."""
        p = Plateau(L=1.0, r=0.1, mean=0.0)
        rng = np.random.default_rng(2)
        samples = p._sample(100_000, rng)
        sigma = samples.std()
        assert 1.0 / (2.0 * math.sqrt(3.0)) <= sigma <= 1.0 / 2.8 + 0.01, \
            f"σ={sigma} outside ASTRA-stated bounds for plateau"

    def test_approaches_uniform_for_sharp_edges(self):
        """As r → 0, plateau should approach uniform σ = L/(2√3)."""
        p = Plateau(L=1.0, r=1e-3, mean=0.0)
        rng = np.random.default_rng(3)
        samples = p._sample(50_000, rng)
        sigma = samples.std()
        uniform_sigma = 1.0 / (2.0 * math.sqrt(3.0))
        assert abs(sigma - uniform_sigma) < 0.01


class TestRadialUniformSampling:
    def test_inside_disk(self):
        ru = RadialUniform(R=1.0)
        rng = np.random.default_rng(0)
        a, b = ru._sample2d(10_000, rng)
        assert a.shape == (10_000,)
        assert b.shape == (10_000,)
        radii = np.sqrt(a * a + b * b)
        assert np.all(radii <= 1.0 + 1e-12)

    def test_uniform_density_on_disk(self):
        """For a uniformly-filled disk of radius R, σ_x = σ_y = R/2."""
        ru = RadialUniform(R=2.0)
        rng = np.random.default_rng(1)
        a, b = ru._sample2d(50_000, rng)
        assert abs(a.std() - 1.0) < 0.02
        assert abs(b.std() - 1.0) < 0.02
        cov = np.mean(a * b) - a.mean() * b.mean()
        assert abs(cov) < 0.02


class TestIsotropicSampling:
    def test_on_sphere_with_forward_pz(self):
        iso = Isotropic(p_mag=1e5)
        rng = np.random.default_rng(0)
        px, py, pz = iso._sample3d(20_000, rng)
        assert px.shape == (20_000,)
        mag = np.sqrt(px ** 2 + py ** 2 + pz ** 2)
        np.testing.assert_allclose(mag, 1e5, rtol=1e-12)
        assert np.all(pz >= -1e-12)

    def test_statistics_match_half_sphere(self):
        """ASTRA: <pz>=P/2, σ_px=σ_py=P/√3, σ_pz=P/(2√3)."""
        P = 1e5
        iso = Isotropic(p_mag=P)
        rng = np.random.default_rng(1)
        px, py, pz = iso._sample3d(100_000, rng)
        assert abs(pz.mean() - P / 2.0) < 0.005 * P
        assert abs(px.std() - P / math.sqrt(3.0)) < 0.005 * P
        assert abs(py.std() - P / math.sqrt(3.0)) < 0.005 * P
        assert abs(pz.std() - P / (2.0 * math.sqrt(3.0))) < 0.005 * P
        assert abs(px.mean()) < 0.01 * P
        assert abs(py.mean()) < 0.01 * P


class TestMakeSliceValidation:
    GAUSS = Gaussian(sig=1e-4)
    GAUSS_PZ = Gaussian(sig=1e3, mean=5e5)

    def test_rejects_zero_n(self):
        with pytest.raises(ValueError, match="n"):
            make_slice(0, I_total=1.0,
                       x=self.GAUSS, y=self.GAUSS,
                       px=self.GAUSS, py=self.GAUSS, pz=self.GAUSS_PZ)

    def test_rejects_nonpositive_I_total(self):
        with pytest.raises(ValueError, match="I_total"):
            make_slice(100, I_total=0.0,
                       x=self.GAUSS, y=self.GAUSS,
                       px=self.GAUSS, py=self.GAUSS, pz=self.GAUSS_PZ)
        with pytest.raises(ValueError, match="I_total"):
            make_slice(100, I_total=-1.0,
                       x=self.GAUSS, y=self.GAUSS,
                       px=self.GAUSS, py=self.GAUSS, pz=self.GAUSS_PZ)

    def test_transverse_conflicts_with_x_or_y(self):
        ru = RadialUniform(R=1e-3)
        with pytest.raises(ValueError, match="transverse"):
            make_slice(100, I_total=1.0, transverse=ru,
                       x=self.GAUSS, y=self.GAUSS,
                       px=self.GAUSS, py=self.GAUSS, pz=self.GAUSS_PZ)

    def test_transverse_momentum_conflicts_with_px_or_py(self):
        ru = RadialUniform(R=1e3)
        with pytest.raises(ValueError, match="transverse_momentum"):
            make_slice(100, I_total=1.0,
                       x=self.GAUSS, y=self.GAUSS,
                       transverse_momentum=ru, px=self.GAUSS,
                       py=self.GAUSS, pz=self.GAUSS_PZ)

    def test_momentum_conflicts_with_px_py_pz_or_transverse_momentum(self):
        iso = Isotropic(p_mag=5e5)
        with pytest.raises(ValueError, match="momentum"):
            make_slice(100, I_total=1.0,
                       x=self.GAUSS, y=self.GAUSS,
                       momentum=iso, pz=self.GAUSS_PZ)

    def test_missing_transverse_position(self):
        with pytest.raises(ValueError, match="x.*y|transverse"):
            make_slice(100, I_total=1.0,
                       px=self.GAUSS, py=self.GAUSS, pz=self.GAUSS_PZ)

    def test_missing_transverse_momentum(self):
        with pytest.raises(ValueError, match="px.*py|transverse_momentum|momentum"):
            make_slice(100, I_total=1.0,
                       x=self.GAUSS, y=self.GAUSS, pz=self.GAUSS_PZ)

    def test_missing_pz(self):
        with pytest.raises(ValueError, match="pz|momentum"):
            make_slice(100, I_total=1.0,
                       x=self.GAUSS, y=self.GAUSS,
                       px=self.GAUSS, py=self.GAUSS)
