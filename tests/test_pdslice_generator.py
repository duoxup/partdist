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
