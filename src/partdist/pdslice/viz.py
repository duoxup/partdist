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
    _to_unit_like,
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
