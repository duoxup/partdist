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
