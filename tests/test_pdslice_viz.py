"""Tests for partdist.pdslice.viz."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from partdist.pdslice.generator import Gaussian, make_slice
from partdist.pdslice import SliceDistribution


def gauss_slice(n: int = 5000, seed: int = 0) -> SliceDistribution:
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
        from partdist.pdslice import viz
        assert viz is not None

    def test_helper_rejects_z(self):
        from partdist.pdslice.viz import _check_z_not_axis
        with pytest.raises(ValueError, match="scalar z"):
            _check_z_not_axis("z")
        with pytest.raises(ValueError, match="scalar z"):
            _check_z_not_axis("x", "z")
        # non-z names pass:
        _check_z_not_axis("x", "y", "px")
