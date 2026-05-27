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


class TestScatterPdslice:
    def test_returns_fig_ax_artist(self):
        from partdist.pdslice.viz import scatter_pdslice
        d = gauss_slice()
        fig, ax, artist = scatter_pdslice(d, x="x", y="px")
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert isinstance(artist, PathCollection)
        plt.close(fig)

    def test_default_c_is_lam_abs(self):
        """Without explicit c, the artist must have a color array (lam_abs)
        and the figure must have a colorbar axis."""
        from partdist.pdslice.viz import scatter_pdslice
        d = gauss_slice()
        fig, ax, artist = scatter_pdslice(d, x="x", y="px")
        # color path engaged: scatter sets get_array() when c= is given
        assert artist.get_array() is not None, \
            "c=None default would yield no per-point color array"
        # colorbar auto-added as a second Axes on the figure
        assert len(fig.axes) >= 2, "expected colorbar axes when c is set"
        plt.close(fig)

    def test_explicit_c_overrides_default(self):
        from partdist.pdslice.viz import scatter_pdslice
        d = gauss_slice()
        fig, ax, artist = scatter_pdslice(d, x="x", y="px", c="pz")
        cb_ax = [a for a in fig.axes if a is not ax][0]
        # colorbar label should reference pz (latex contains 'p_z')
        label = cb_ax.get_ylabel() or cb_ax.get_xlabel()
        assert "p_z" in label or "pz" in label
        plt.close(fig)

    def test_rejects_x_z(self):
        from partdist.pdslice.viz import scatter_pdslice
        d = gauss_slice()
        with pytest.raises(ValueError, match="scalar z"):
            scatter_pdslice(d, x="z", y="px")

    def test_rejects_y_z(self):
        from partdist.pdslice.viz import scatter_pdslice
        d = gauss_slice()
        with pytest.raises(ValueError, match="scalar z"):
            scatter_pdslice(d, x="x", y="z")
