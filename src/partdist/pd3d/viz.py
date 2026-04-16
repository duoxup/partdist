from __future__ import annotations

from typing import Any, Optional, Sequence, Union

import numpy as np
import xtils

from .core import ParticleDistribution


def _apply_prefix_to_unit(unit: str, prefix: str) -> str:
    """
    Apply an SI prefix to a unit string.

    Examples
    --------
    'm'    + 'm' -> 'mm'
    's'    + 'n' -> 'ns'
    'eV/c' + 'k' -> 'keV/c'
    ''     + any -> ''
    """
    unit = unit or ""
    if not unit:
        return ""
    return f"{prefix}{unit}"


def _label_from_key(
    dist: ParticleDistribution,
    key: str,
    *,
    unit_override: str | None = None,
) -> str:
    """
    Build an axis/colorbar label from quantity metadata.
    """
    q = dist.get_quantity(key)
    name = q.latex_name or q.short_name or q.name
    unit = q.unit if unit_override is None else unit_override

    if unit:
        return f"{name} [{unit}]"
    return f"{name}"


def _autoscale_data_and_unit(
    data: np.ndarray,
    unit: str,
) -> tuple[np.ndarray, str]:
    """
    Autoscale data using xtils.get_autoscale and update the unit string.

    Returns
    -------
    scaled_data, scaled_unit
    """
    if (unit is not None) and unit != '':
        scale, prefix = xtils.get_autoscale(data)
    else:
        scale, prefix = 1, ''
    scaled_data = np.asarray(data, dtype=float) * scale
    scaled_unit = _apply_prefix_to_unit(unit, prefix)
    return scaled_data, scaled_unit


def _resolve_range(
    x: np.ndarray,
    y: np.ndarray,
    range_: Sequence[Sequence[float]] | None = None,
    xrange: Sequence[float] | None = None,
    yrange: Sequence[float] | None = None,
) -> list[list[float]] | None:
    """
    Resolve histogram range specification.

    Priority:
    1. range_
    2. xrange + yrange
    3. None
    """
    if range_ is not None:
        if len(range_) != 2:
            raise ValueError("range must be ((xmin, xmax), (ymin, ymax)).")
        return [list(range_[0]), list(range_[1])]

    if xrange is None and yrange is None:
        return None

    if xrange is None:
        xrange = (float(np.min(x)), float(np.max(x)))
    if yrange is None:
        yrange = (float(np.min(y)), float(np.max(y)))

    return [list(xrange), list(yrange)]


def scatter_pd3d(
    dist: ParticleDistribution,
    *,
    x: str,
    y: str,
    c: str | None = None,
    fig=None,
    ax=None,
    colorbar: bool | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    clabel: str | None = None,
    **scatter_kwargs: Any,
):
    """
    Plot a scatter plot from ParticleDistribution quantities.

    Parameters
    ----------
    dist
        ParticleDistribution instance.
    x, y
        Quantity keys for x and y axes.
    c
        Optional quantity key for color mapping.
    fig, ax
        Optional matplotlib figure and axes. They are normalized by
        `xtils.ensure_fig_ax(fig, ax)`.
    colorbar
        Whether to add a colorbar when `c` is not None.
        Default:
        - True if c is given
        - False otherwise
    xlabel, ylabel, clabel
        Optional manual labels. If omitted, labels are built from quantity metadata.
    **scatter_kwargs
        Forwarded to `ax.scatter(...)`.

    Returns
    -------
    fig, ax, artist
        Matplotlib figure, axes, and scatter artist.
    """
    fig, ax = xtils.ensure_fig_ax(fig, ax)
    fig.set_layout_engine('constrained')

    xq = dist.get_quantity(x)
    yq = dist.get_quantity(y)

    xdata_raw = np.asarray(xq.data, dtype=float)
    ydata_raw = np.asarray(yq.data, dtype=float)

    xdata, xunit_scaled = _autoscale_data_and_unit(xdata_raw, xq.unit)
    ydata, yunit_scaled = _autoscale_data_and_unit(ydata_raw, yq.unit)

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


def hist2d_pd3d(
    dist: ParticleDistribution,
    *,
    x: str,
    y: str,
    weight: str | None = 'Q_abs',
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
    """
    Plot a 2D histogram from ParticleDistribution quantities.

    Parameters
    ----------
    dist
        ParticleDistribution instance.
    x, y
        Quantity keys for x and y axes.
    weight
        Optional quantity key used as histogram weights.
        If None, ordinary counts are used.
    bins
        Histogram bins, passed to `numpy.histogram2d`.
    range
        Histogram range in the form ((xmin, xmax), (ymin, ymax)).
    xrange, yrange
        Optional separate x/y ranges. Ignored if `range` is provided.
    color_threshold
        Optional threshold in relative units.
        Bins with value < color_threshold * max(hist) are masked and not colored.

        Example:
        - color_threshold=0.01 means bins below 1% of the maximum bin value are hidden.
    fig, ax
        Optional matplotlib figure and axes. They are normalized by
        `xtils.ensure_fig_ax(fig, ax)`.
    colorbar
        Whether to add a colorbar.
    xlabel, ylabel, clabel
        Optional manual labels.
    **pcolormesh_kwargs
        Extra keyword arguments passed to `ax.pcolormesh(...)`.

    Returns
    -------
    fig, ax, mesh, hist, xedges, yedges
        Matplotlib figure, axes, QuadMesh artist, histogram array,
        and bin edges.
    """
    fig, ax = xtils.ensure_fig_ax(fig, ax)
    fig.set_layout_engine('constrained')

    xq = dist.get_quantity(x)
    yq = dist.get_quantity(y)

    xdata_raw = np.asarray(xq.data, dtype=float)
    ydata_raw = np.asarray(yq.data, dtype=float)

    hist_range_raw = _resolve_range(
        xdata_raw,
        ydata_raw,
        range_=range,
        xrange=xrange,
        yrange=yrange,
    )

    weights = None
    if weight is not None:
        weights = np.asarray(dist.get_data(weight), dtype=float)

    hist, xedges_raw, yedges_raw = np.histogram2d(
        xdata_raw,
        ydata_raw,
        bins=bins,
        range=hist_range_raw,
        weights=weights,
    )

    # Autoscale x/y bin edges for plotting.
    _, xunit_scaled = _autoscale_data_and_unit(xdata_raw, xq.unit)
    _, yunit_scaled = _autoscale_data_and_unit(ydata_raw, yq.unit)
    
    if (xq.unit is not None) and (xq.unit != ''):
        xscale, _ = xtils.get_autoscale(xdata_raw)
    else:
        xscale = 1
        
    if (yq.unit is not None) and (yq.unit != ''):
        yscale, _ = xtils.get_autoscale(ydata_raw)
    else:
        yscale = 1

    xedges = np.asarray(xedges_raw, dtype=float) * xscale
    yedges = np.asarray(yedges_raw, dtype=float) * yscale

    # numpy.histogram2d returns shape (nx, ny), so transpose for pcolormesh.
    hist_plot = hist.T

    if color_threshold is not None:
        if not (0.0 <= color_threshold <= 1.0):
            raise ValueError("color_threshold must be between 0 and 1.")
        hmax = np.nanmax(hist_plot)
        if hmax > 0.0:
            hist_plot = np.ma.masked_less(hist_plot, color_threshold * hmax)

    if "shading" not in pcolormesh_kwargs:
        pcolormesh_kwargs["shading"] = "auto"

    mesh = ax.pcolormesh(
        xedges,
        yedges,
        hist_plot,
        **pcolormesh_kwargs,
    )

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
        _add_projection_curves_pd3d(dist, ax=ax, x=x, y=y,
                                    weight=weight,
                                    normalize=True,
                                    xproj_scale=0.1,
                                    yproj_scale=0.1,
                                    xproj_kwargs={'color':'red',
                                                  'lw': 1,
                                                  },
                                    yproj_kwargs={'color':'red',
                                                  'lw': 1,
                                                  },
                                    show_xproj_axis=False,
                                    show_yproj_axis=False,
                                    )

    return fig, ax, mesh, hist, xedges_raw, yedges_raw


def _add_projection_curves_pd3d(
    dist: ParticleDistribution,
    *,
    ax,
    x: str,
    y: str,
    weight: str | None = 'Q_abs',
    bins: int | Sequence[int] | Sequence[np.ndarray] = 100,
    range: Sequence[Sequence[float]] | None = None,
    xrange: Sequence[float] | None = None,
    yrange: Sequence[float] | None = None,
    normalize: bool = False,
    xproj_scale: float = 1.0,
    yproj_scale: float = 1.0,
    xproj_kwargs: dict[str, Any] | None = None,
    yproj_kwargs: dict[str, Any] | None = None,
    show_xproj_axis: bool = True,
    show_yproj_axis: bool = True,
):
    """
    Add x/y projection curves to an existing 2D phase-space axes.

    The x projection is drawn on a secondary y-axis sharing the same x-axis.
    The y projection is drawn on a secondary x-axis sharing the same y-axis.

    Parameters
    ----------
    dist
        ParticleDistribution instance.
    ax
        Existing matplotlib axes that already contains the 2D histogram.
    x, y
        Quantity keys for x and y axes.
    weight
        Optional quantity key used as histogram weights.
        If None, ordinary counts are used.
    bins
        Histogram bins, passed to `numpy.histogram2d`.
        Should be chosen consistently with `hist2d_pd3d`.
    range
        Histogram range in the form ((xmin, xmax), (ymin, ymax)).
    xrange, yrange
        Optional separate x/y ranges. Ignored if `range` is provided.
    normalize
        If True, normalize each projection to its own maximum.
    xproj_scale, yproj_scale
        Additional multiplicative scale factors applied after optional
        normalization. Useful for visually tuning the projection amplitudes.
    xproj_kwargs, yproj_kwargs
        Extra keyword arguments passed to matplotlib `plot(...)`.
    show_xproj_axis, show_yproj_axis
        Whether to show the secondary axes ticks/labels for the projections.

    Returns
    -------
    ax_xproj, ax_yproj, line_xproj, line_yproj, proj_x, proj_y, xcenters_raw, ycenters_raw
        Secondary axes, line artists, projection arrays, and raw bin centers.
    """
    xproj_kwargs = {} if xproj_kwargs is None else dict(xproj_kwargs)
    yproj_kwargs = {} if yproj_kwargs is None else dict(yproj_kwargs)

    xq = dist.get_quantity(x)
    yq = dist.get_quantity(y)

    xdata_raw = np.asarray(xq.data, dtype=float)
    ydata_raw = np.asarray(yq.data, dtype=float)

    hist_range_raw = _resolve_range(
        xdata_raw,
        ydata_raw,
        range_=range,
        xrange=xrange,
        yrange=yrange,
    )

    weights = None
    if weight is not None:
        weights = np.asarray(dist.get_data(weight), dtype=float)

    hist, xedges_raw, yedges_raw = np.histogram2d(
        xdata_raw,
        ydata_raw,
        bins=bins,
        range=hist_range_raw,
        weights=weights,
    )

    proj_x = hist.sum(axis=1)
    proj_y = hist.sum(axis=0)

    xcenters_raw = 0.5 * (xedges_raw[:-1] + xedges_raw[1:])
    ycenters_raw = 0.5 * (yedges_raw[:-1] + yedges_raw[1:])

    if (xq.unit is not None) and (xq.unit != ""):
        xscale, _ = xtils.get_autoscale(xdata_raw)
    else:
        xscale = 1.0

    if (yq.unit is not None) and (yq.unit != ""):
        yscale, _ = xtils.get_autoscale(ydata_raw)
    else:
        yscale = 1.0

    xcenters = np.asarray(xcenters_raw, dtype=float) * xscale
    ycenters = np.asarray(ycenters_raw, dtype=float) * yscale

    proj_x_plot = np.asarray(proj_x, dtype=float)
    proj_y_plot = np.asarray(proj_y, dtype=float)

    if normalize:
        if np.nanmax(proj_x_plot) > 0.0:
            proj_x_plot = proj_x_plot / np.nanmax(proj_x_plot)
        if np.nanmax(proj_y_plot) > 0.0:
            proj_y_plot = proj_y_plot / np.nanmax(proj_y_plot)

    proj_x_plot *= xproj_scale
    proj_y_plot *= yproj_scale

    ax_xproj = ax.twinx()
    ax_yproj = ax.twiny()

    if "clip_on" not in xproj_kwargs:
        xproj_kwargs["clip_on"] = True
    if "clip_on" not in yproj_kwargs:
        yproj_kwargs["clip_on"] = True

    line_xproj = ax_xproj.plot(xcenters, proj_x_plot, **xproj_kwargs)[0]
    line_yproj = ax_yproj.plot(proj_y_plot, ycenters, **yproj_kwargs)[0]

    ax_xproj.set_xlim(ax.get_xlim())
    ax_yproj.set_ylim(ax.get_ylim())
    
    if normalize:
        ax_xproj.set_ylim([0, 1])
        ax_yproj.set_xlim([0, 1])

    if not show_xproj_axis:
        ax_xproj.set_yticks([])
        ax_xproj.set_ylabel("")
        ax_xproj.spines["right"].set_visible(False)

    if not show_yproj_axis:
        ax_yproj.set_xticks([])
        ax_yproj.set_xlabel("")
        ax_yproj.spines["top"].set_visible(False)

    return (
        ax_xproj,
        ax_yproj,
        line_xproj,
        line_yproj,
        proj_x,
        proj_y,
        xcenters_raw,
        ycenters_raw,
    )


def plot_binned_profile(
    dist: ParticleDistribution,
    x: Union[str, np.ndarray],
    y: Union[str, np.ndarray],
    *,
    bins: int = 100,
    x_range: Optional[tuple[float, float]] = None,
    weight: Union[None, str, np.ndarray] = "Q_abs",
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
    """
    Compute and plot a 1D binned profile y(x).

    Typical use cases
    -----------------
    Slice centroid vs z::

        plot_binned_profile(dist, x="z", y="x")

    Slice energy spread vs z::

        plot_binned_profile(dist, x="z", y="pz", plot_stat="std")

    Parameters
    ----------
    dist
        ParticleDistribution instance.
    x, y
        Quantity keys (str) or explicit arrays. Labels and unit autoscaling
        are applied automatically when string keys are given.
    bins
        Number of bins.
    x_range
        Explicit x range ``(xmin, xmax)``. If None, inferred from data.
    weight
        Particle weights passed to ``compute_binned_profile``.
    stat
        Binned statistic for ``y_stat``: ``"mean"``, ``"weighted_mean"``,
        or ``"median"``.
    min_count, min_weight_sum
        Validity thresholds for bins.
    mask
        Optional particle mask.
    plot_stat
        Which per-bin value to draw:

        - ``"stat"``  – the main statistic (default)
        - ``"std"``   – per-bin standard deviation
        - ``"rms"``   – per-bin RMS

    show_std_band
        If ``True`` and ``plot_stat="stat"``, shade ±σ around the line.
        Ignored for other ``plot_stat`` values.
    std_band_alpha
        Alpha of the shaded band.
    fig, ax
        Optional matplotlib figure and axes.
    xlabel, ylabel
        Override axis labels. Auto-generated from quantity metadata if None.
    **plot_kwargs
        Forwarded to ``ax.plot(...)``.

    Returns
    -------
    fig, ax, line, profile
        Matplotlib figure, axes, Line2D artist, and the
        :class:`~partdist.pd3d.analysis.BinnedProfileResult`.
    """
    from .analysis import compute_binned_profile

    fig, ax = xtils.ensure_fig_ax(fig, ax)
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

    # Select y values to plot
    if plot_stat == "stat":
        y_data = profile.y_valid
    elif plot_stat == "std":
        y_data = profile.y_std_valid
    elif plot_stat == "rms":
        y_data = profile.y_rms_valid
    else:
        raise ValueError(f"Unknown plot_stat={plot_stat!r}. Choose 'stat', 'std', or 'rms'.")

    x_data = profile.x_valid

    # --- unit / autoscale ---
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

    x_data_scaled, x_unit_scaled = _autoscale_data_and_unit(x_data, x_unit)
    y_data_scaled, y_unit_scaled = _autoscale_data_and_unit(y_data, y_unit)

    # --- plot ---
    line, = ax.plot(x_data_scaled, y_data_scaled, **plot_kwargs)

    # Optional ±σ band (only meaningful for the main statistic)
    if show_std_band and plot_stat == "stat":
        y_std = profile.y_std_valid
        if y_unit and len(y_data) > 0:
            yscale, _ = xtils.get_autoscale(y_data)
        else:
            yscale = 1.0
        y_std_scaled = y_std * yscale
        ax.fill_between(
            x_data_scaled,
            y_data_scaled - y_std_scaled,
            y_data_scaled + y_std_scaled,
            alpha=std_band_alpha,
            color=line.get_color(),
        )

    # --- axis labels ---
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