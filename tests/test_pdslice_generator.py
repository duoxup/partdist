"""Tests for partdist.pdslice.generator."""
from __future__ import annotations

import math
import numpy as np
import pytest

from partdist.pdslice.generator import (
    Gaussian, Uniform, Plateau, RadialUniform, Isotropic,
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
