"""
I/O routines for SliceDistribution.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from scipy.constants import c as g_c

from ..particle_array_quantity import ParticleArrayQuantity
from .core import SliceDistribution


CST_PID_N_COLS = 9  # pos_x pos_y pos_z mom_x mom_y mom_z mass charge current


def _loadtxt_pid(filepath: str | Path, dtype: type = float) -> np.ndarray:
    """Load a CST .pid 9-column ASCII table with '%' line comments."""
    arr = np.loadtxt(Path(filepath), dtype=dtype, ndmin=2, comments="%")
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {arr.shape}.")
    if arr.shape[1] != CST_PID_N_COLS:
        raise ValueError(
            f"CST .pid file must have exactly {CST_PID_N_COLS} columns, "
            f"got {arr.shape[1]}."
        )
    if arr.shape[0] < 1:
        raise ValueError("CST .pid file must contain at least one row.")
    return arr


def read_cst_pid_distribution(
    filepath: str | Path,
    *,
    fixed_axis: Literal["x", "y", "z"] = "z",
    fixed_value: float | None = None,
    dtype: type = float,
) -> SliceDistribution:
    """
    Read a CST Particle Studio ``.pid`` file into a :class:`SliceDistribution`.

    File format (SI units, ``%`` line comments)
    -------------------------------------------
    Columns: ``pos_x  pos_y  pos_z  mom_x  mom_y  mom_z  mass  charge  current``

    where ``mom_i`` is the normalised momentum :math:`\\beta_i \\gamma_i`
    (dimensionless), ``mass`` [kg] and ``charge`` [C, signed] are the
    per-particle physical species values, and ``current`` [A] is the current
    contribution of each macroparticle through the chosen cross-section.

    A .pid file describes a **DC (steady-state) particle distribution**
    crossing a fixed plane, so :class:`SliceDistribution` is its natural
    container.  The macroparticle current is converted to a linear charge
    density via ``lam = current / v_d``, where ``v_d`` is the velocity
    component along the fixed axis — this is the standard relation between
    line charge density and current for steady flow through a cross-section.

    Conversion
    ----------
    - Two varying position columns become the slice's transverse positions.
    - The fixed-axis column is collapsed to a single ``fixed_value`` (mean
      across rows when the caller does not specify one).
    - px, py, pz: ``mom * mass * c**2 / |charge|`` [eV/c], computed per row
      so heterogeneous species work.
    - t          : zeros (the format carries no time information).
    - lam        : ``current / v_d`` [C/m], with sign preserved.
    - extras     : raw ``current`` [A], ``mass`` [kg] and ``charge`` [C] are
      preserved as ``cst_current`` / ``cst_mass`` / ``cst_charge`` so no
      information is lost.

    Parameters
    ----------
    filepath
        Path to the .pid file.
    fixed_axis
        Axis perpendicular to the slice plane.  Defaults to ``'z'`` (the
        usual beam-propagation direction).
    fixed_value
        Position of the slice plane along ``fixed_axis`` [m].  When ``None``
        (default), the mean of the corresponding column from the file is
        used.
    dtype
        Data type passed to ``np.loadtxt``.
    """
    if fixed_axis not in ("x", "y", "z"):
        raise ValueError(f"fixed_axis must be 'x', 'y', or 'z'; got {fixed_axis!r}.")

    raw = _loadtxt_pid(filepath, dtype=dtype)

    pos = {"x": raw[:, 0], "y": raw[:, 1], "z": raw[:, 2]}
    p_norm = {"x": raw[:, 3], "y": raw[:, 4], "z": raw[:, 5]}
    mass = raw[:, 6]
    charge = raw[:, 7]
    current = raw[:, 8].astype(float)

    n = raw.shape[0]

    # Per-row rest energy m c^2 / |q|, used to convert mom (= beta*gamma) to eV/c.
    rest_energy_eV = mass * g_c**2 / np.abs(charge)
    p_eVc = {axis: p_norm[axis] * rest_energy_eV for axis in ("x", "y", "z")}

    # gamma from |p_norm|^2 = (beta*gamma)^2 = gamma^2 - 1.
    p_norm_sq = p_norm["x"] ** 2 + p_norm["y"] ** 2 + p_norm["z"] ** 2
    gamma = np.sqrt(1.0 + p_norm_sq)
    v_d = (p_norm[fixed_axis] / gamma) * g_c  # [m/s] along the fixed axis
    lam = current / v_d                       # [C/m], sign-preserving

    if fixed_value is None:
        fixed_value = float(np.mean(pos[fixed_axis]))

    varying = {"x": ("y", "z"), "y": ("x", "z"), "z": ("x", "y")}[fixed_axis]
    pos_kwargs = {axis: pos[axis] for axis in varying}

    extras = {
        "cst_current": ParticleArrayQuantity(
            name="cst_current",
            data=current,
            unit="A",
            dtype_kind="float",
            short_name="I_cst",
            long_name="CST .pid macroparticle current",
            latex_name=r"$I_\mathrm{cst}$",
            category="current",
            is_derived=False,
        ),
        "cst_mass": ParticleArrayQuantity(
            name="cst_mass",
            data=mass.astype(float),
            unit="kg",
            dtype_kind="float",
            short_name="m_cst",
            long_name="CST .pid per-particle rest mass",
            latex_name=r"$m_\mathrm{cst}$",
            category="other",
            is_derived=False,
        ),
        "cst_charge": ParticleArrayQuantity(
            name="cst_charge",
            data=charge.astype(float),
            unit="C",
            dtype_kind="float",
            short_name="q_cst",
            long_name="CST .pid per-particle physical charge",
            latex_name=r"$q_\mathrm{cst}$",
            category="other",
            is_derived=False,
        ),
    }

    return SliceDistribution(
        fixed_axis=fixed_axis,
        fixed_value=fixed_value,
        **pos_kwargs,
        px=p_eVc["x"], py=p_eVc["y"], pz=p_eVc["z"],
        t=np.zeros(n, dtype=float),
        lam=lam,
        extras=extras,
    )
