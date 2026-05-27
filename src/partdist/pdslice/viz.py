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
