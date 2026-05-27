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


class TestHist2dPdslice:
    def test_returns_six_tuple(self):
        from partdist.pdslice.viz import hist2d_pdslice
        d = gauss_slice()
        result = hist2d_pdslice(d, x="x", y="px", show_projections=False)
        assert len(result) == 6
        fig, ax, mesh, hist, xedges, yedges = result
        assert hist.shape == (100, 100)
        plt.close(fig)

    def test_default_weight_is_lam_abs(self):
        """For uniform-lam fixture, lam_abs-weighted and unweighted hists
        differ only by a scalar multiple — normalised forms match."""
        from partdist.pdslice.viz import hist2d_pdslice
        d = gauss_slice()
        fig_c, _, _, hist_counts, _, _ = hist2d_pdslice(
            d, x="x", y="px", weight=None, show_projections=False,
        )
        plt.close(fig_c)
        fig_l, _, _, hist_lam, _, _ = hist2d_pdslice(
            d, x="x", y="px", show_projections=False,
        )
        plt.close(fig_l)
        nz = hist_counts > 0
        ratio_counts = hist_counts[nz] / hist_counts[nz].mean()
        ratio_lam = hist_lam[nz] / hist_lam[nz].mean()
        np.testing.assert_allclose(ratio_lam, ratio_counts, rtol=1e-12)

    def test_weight_none_gives_counts(self):
        from partdist.pdslice.viz import hist2d_pdslice
        d = gauss_slice()
        fig, _, _, hist, _, _ = hist2d_pdslice(
            d, x="x", y="px", weight=None, show_projections=False,
        )
        plt.close(fig)
        assert int(hist.sum()) == len(d)

    def test_show_projections_off(self):
        from partdist.pdslice.viz import hist2d_pdslice
        d = gauss_slice()
        fig, ax, _, _, _, _ = hist2d_pdslice(
            d, x="x", y="px", show_projections=False,
        )
        assert len(ax.lines) == 0
        plt.close(fig)

    def test_rejects_z_axis(self):
        from partdist.pdslice.viz import hist2d_pdslice
        d = gauss_slice()
        with pytest.raises(ValueError, match="scalar z"):
            hist2d_pdslice(d, x="z", y="px")
        with pytest.raises(ValueError, match="scalar z"):
            hist2d_pdslice(d, x="x", y="z")


class TestPlotBinnedProfile:
    def test_returns_four_tuple(self):
        from partdist.pdslice.viz import plot_binned_profile
        d = gauss_slice()
        fig, ax, line, profile = plot_binned_profile(d, "x", "px")
        assert isinstance(line, Line2D)
        from partdist.pd3d.analysis import BinnedProfileResult
        assert isinstance(profile, BinnedProfileResult)
        plt.close(fig)

    def test_default_weight_calls_lam_abs(self):
        """If the default were 'Q_abs' it would KeyError on a SliceDistribution
        (no Q_abs derived quantity). Reaching weight_sum>0 proves lam_abs."""
        from partdist.pdslice.viz import plot_binned_profile
        d = gauss_slice()
        fig, ax, line, profile = plot_binned_profile(d, "x", "px")
        assert float(np.sum(profile.weight_sum)) > 0
        plt.close(fig)

    def test_explicit_array_skips_z_check(self):
        from partdist.pdslice.viz import plot_binned_profile
        d = gauss_slice()
        x_arr = d.get_data("x")
        y_arr = d.get_data("px")
        fig, ax, line, profile = plot_binned_profile(d, x_arr, y_arr)
        # no exception; runs to completion
        from partdist.pd3d.analysis import BinnedProfileResult
        assert isinstance(profile, BinnedProfileResult)
        plt.close(fig)

    def test_rejects_x_key_z(self):
        from partdist.pdslice.viz import plot_binned_profile
        d = gauss_slice()
        with pytest.raises(ValueError, match="scalar z"):
            plot_binned_profile(d, "z", "px")
