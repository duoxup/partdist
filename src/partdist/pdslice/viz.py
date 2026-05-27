"""Matplotlib-based viz for SliceDistribution.

Sibling of partdist.pd3d.viz, restricted to operations meaningful on a
single transverse slice. The slice has scalar z (all particles share
one z0), so any axis spec involving 'z' raises ValueError.

Default weight is 'lam_abs' (the slice analog of pd3d's 'Q_abs').
scatter_pdslice also defaults its color-mapping quantity 'c' to
'lam_abs' so per-particle charge density is the default visual
encoding.

Private helpers are imported from partdist.pd3d.viz to avoid
duplication; the pd3d helpers depend only on the get_quantity /
get_data / get_quantity_* metadata accessors that SliceDistribution
also provides.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence, Union

import numpy as np

from ..pd3d.viz import (
    _ensure_fig_ax,
    _autoscale,
    _label_from_key,
    _resolve_range,
    _add_projection_curves_pd3d,
)
from .core import SliceDistribution


def _check_z_not_axis(*args: str) -> None:
    """Raise if any argument equals 'z'.

    SliceDistribution has scalar z; binning, scattering, or computing
    second moments along z is degenerate by construction.
    """
    for k in args:
        if k == "z":
            raise ValueError(
                "SliceDistribution has scalar z (all particles share one z0); "
                "use a transverse or momentum axis instead."
            )


def scatter_pdslice(
    dist: SliceDistribution,
    *,
    x: str,
    y: str,
    c: str | None = "lam_abs",
    fig=None,
    ax=None,
    colorbar: bool | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    clabel: str | None = None,
    **scatter_kwargs: Any,
):
    """Scatter plot for SliceDistribution.

    Mirrors partdist.pd3d.viz.scatter_pd3d but (a) rejects 'z' axis keys
    and (b) defaults c='lam_abs' so per-particle charge density is the
    default visual encoding. Pass c=None for a plain (uncoloured) scatter.

    Returns (fig, ax, artist).
    """
    _check_z_not_axis(x, y)

    fig, ax = _ensure_fig_ax(fig, ax)
    fig.set_layout_engine("constrained")

    xq = dist.get_quantity(x)
    yq = dist.get_quantity(y)

    xdata_raw = np.asarray(xq.data, dtype=float)
    ydata_raw = np.asarray(yq.data, dtype=float)

    xscale, xunit_scaled = _autoscale(xdata_raw, xq.unit)
    yscale, yunit_scaled = _autoscale(ydata_raw, yq.unit)
    xdata = xdata_raw * xscale
    ydata = ydata_raw * yscale

    if c is None:
        artist = ax.scatter(xdata, ydata, **scatter_kwargs)
    else:
        cdata = np.asarray(dist.get_data(c))
        artist = ax.scatter(xdata, ydata, c=cdata, **scatter_kwargs)

    ax.set_xlabel(_label_from_key(dist, x, unit_override=xunit_scaled) if xlabel is None else xlabel)
    ax.set_ylabel(_label_from_key(dist, y, unit_override=yunit_scaled) if ylabel is None else ylabel)

    if c is not None:
        if colorbar is None:
            colorbar = True
        if colorbar:
            cb = fig.colorbar(artist, ax=ax)
            cb.set_label(_label_from_key(dist, c) if clabel is None else clabel)

    return fig, ax, artist


def hist2d_pdslice(
    dist: SliceDistribution,
    *,
    x: str,
    y: str,
    weight: str | None = "lam_abs",
    bins: int | Sequence[int] | Sequence[np.ndarray] = 100,
    range: Sequence[Sequence[float]] | None = None,
    xrange: Sequence[float] | None = None,
    yrange: Sequence[float] | None = None,
    color_threshold: float | None = None,
    fig=None,
    ax=None,
    colorbar: bool = True,
    xlabel: str | None = None,
    ylabel: str | None = None,
    clabel: str | None = None,
    show_projections: bool = True,
    **pcolormesh_kwargs: Any,
):
    """2D histogram for SliceDistribution.

    Mirrors partdist.pd3d.viz.hist2d_pd3d but (a) rejects 'z' axis keys
    and (b) defaults weight='lam_abs' instead of 'Q_abs'.

    Returns (fig, ax, mesh, hist, xedges, yedges).
    """
    _check_z_not_axis(x, y)

    fig, ax = _ensure_fig_ax(fig, ax)
    fig.set_layout_engine("constrained")

    xq = dist.get_quantity(x)
    yq = dist.get_quantity(y)

    xdata_raw = np.asarray(xq.data, dtype=float)
    ydata_raw = np.asarray(yq.data, dtype=float)

    hist_range_raw = _resolve_range(
        xdata_raw, ydata_raw,
        range_=range, xrange=xrange, yrange=yrange,
    )

    weights = None
    if weight is not None:
        weights = np.asarray(dist.get_data(weight), dtype=float)

    hist, xedges_raw, yedges_raw = np.histogram2d(
        xdata_raw, ydata_raw,
        bins=bins, range=hist_range_raw, weights=weights,
    )

    xscale, xunit_scaled = _autoscale(xdata_raw, xq.unit)
    yscale, yunit_scaled = _autoscale(ydata_raw, yq.unit)

    xedges = np.asarray(xedges_raw, dtype=float) * xscale
    yedges = np.asarray(yedges_raw, dtype=float) * yscale

    hist_plot = hist.T

    if color_threshold is not None:
        if not (0.0 <= color_threshold <= 1.0):
            raise ValueError("color_threshold must be between 0 and 1.")
        hmax = np.nanmax(hist_plot)
        if hmax > 0.0:
            hist_plot = np.ma.masked_less(hist_plot, color_threshold * hmax)

    if "shading" not in pcolormesh_kwargs:
        pcolormesh_kwargs["shading"] = "auto"

    mesh = ax.pcolormesh(xedges, yedges, hist_plot, **pcolormesh_kwargs)

    x_pad = 0.15 * (xedges[-1] - xedges[0])
    y_pad = 0.15 * (yedges[-1] - yedges[0])
    ax.set_xlim(xedges[0] - x_pad, xedges[-1] + x_pad)
    ax.set_ylim(yedges[0] - y_pad, yedges[-1] + y_pad)

    ax.set_xlabel(_label_from_key(dist, x, unit_override=xunit_scaled) if xlabel is None else xlabel)
    ax.set_ylabel(_label_from_key(dist, y, unit_override=yunit_scaled) if ylabel is None else ylabel)

    if colorbar:
        cb = fig.colorbar(mesh, ax=ax)
        if clabel is None:
            if weight is None:
                clabel = "Counts"
            else:
                clabel = _label_from_key(dist, weight)
        cb.set_label(clabel)

    if show_projections:
        _add_projection_curves_pd3d(
            dist, ax=ax, x=x, y=y, weight=weight,
            normalize=True,
            xproj_scale=0.1, yproj_scale=0.1,
            xproj_kwargs={"color": "red", "lw": 1},
            yproj_kwargs={"color": "red", "lw": 1},
            show_xproj_axis=False, show_yproj_axis=False,
        )

    return fig, ax, mesh, hist, xedges_raw, yedges_raw


def plot_binned_profile(
    dist: SliceDistribution,
    x: Union[str, np.ndarray],
    y: Union[str, np.ndarray],
    *,
    bins: int = 100,
    x_range: Optional[tuple[float, float]] = None,
    weight: Union[None, str, np.ndarray] = "lam_abs",
    stat: str = "weighted_mean",
    min_count: int = 1,
    min_weight_sum: float = 0.0,
    mask: Optional[np.ndarray] = None,
    plot_stat: str = "stat",
    show_std_band: bool = False,
    std_band_alpha: float = 0.3,
    fig=None,
    ax=None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    **plot_kwargs: Any,
):
    """1D binned profile y(x) for SliceDistribution.

    Mirrors partdist.pd3d.viz.plot_binned_profile but (a) rejects 'z' axis
    keys (string only — ndarray inputs skip the check) and (b) defaults
    weight='lam_abs' instead of 'Q_abs'. The latter matters because
    SliceDistribution has no 'Q_abs' derived quantity; using the upstream
    pd3d default would raise KeyError.

    Returns (fig, ax, line, profile).
    """
    if isinstance(x, str):
        _check_z_not_axis(x)
    if isinstance(y, str):
        _check_z_not_axis(y)

    from ..pd3d.analysis import compute_binned_profile

    fig, ax = _ensure_fig_ax(fig, ax)
    fig.set_layout_engine("constrained")

    profile = compute_binned_profile(
        dist,
        x,
        y,
        bins=bins,
        x_range=x_range,
        weight=weight,
        stat=stat,
        min_count=min_count,
        min_weight_sum=min_weight_sum,
        mask=mask,
    )

    if plot_stat == "stat":
        y_data = profile.y_valid
    elif plot_stat == "std":
        y_data = profile.y_std_valid
    elif plot_stat == "rms":
        y_data = profile.y_rms_valid
    else:
        raise ValueError(f"Unknown plot_stat={plot_stat!r}. Choose 'stat', 'std', or 'rms'.")

    x_data = profile.x_valid

    x_unit = ""
    if isinstance(x, str):
        try:
            x_unit = dist.get_quantity(x).unit or ""
        except KeyError:
            pass

    y_unit = ""
    if isinstance(y, str):
        try:
            y_unit = dist.get_quantity(y).unit or ""
        except KeyError:
            pass

    x_mult, x_unit_scaled = _autoscale(x_data, x_unit)
    y_mult, y_unit_scaled = _autoscale(y_data, y_unit)
    x_data_scaled = np.asarray(x_data, dtype=float) * x_mult
    y_data_scaled = np.asarray(y_data, dtype=float) * y_mult

    line, = ax.plot(x_data_scaled, y_data_scaled, **plot_kwargs)

    if show_std_band and plot_stat == "stat":
        y_std_scaled = np.asarray(profile.y_std_valid, dtype=float) * y_mult
        ax.fill_between(
            x_data_scaled,
            y_data_scaled - y_std_scaled,
            y_data_scaled + y_std_scaled,
            alpha=std_band_alpha,
            color=line.get_color(),
        )

    if xlabel is None and isinstance(x, str):
        try:
            xlabel = _label_from_key(dist, x, unit_override=x_unit_scaled)
        except KeyError:
            xlabel = str(x)
    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if ylabel is None and isinstance(y, str):
        try:
            yq = dist.get_quantity(y)
            yname = yq.latex_name or yq.short_name or yq.name
            if plot_stat == "std":
                ylabel = (
                    f"σ({yname}) [{y_unit_scaled}]" if y_unit_scaled else f"σ({yname})"
                )
            elif plot_stat == "rms":
                ylabel = (
                    f"RMS({yname}) [{y_unit_scaled}]" if y_unit_scaled else f"RMS({yname})"
                )
            else:
                ylabel = _label_from_key(dist, y, unit_override=y_unit_scaled)
        except KeyError:
            ylabel = str(y)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    return fig, ax, line, profile
