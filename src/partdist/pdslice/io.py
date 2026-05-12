"""
I/O routines for SliceDistribution.
"""
from __future__ import annotations

from pathlib import Path

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
    z: float | None = None,
    plane_tol: float = 1e-9,
    dtype: type = float,
) -> SliceDistribution:
    """
    Read a CST Particle Studio ``.pid`` file into a :class:`SliceDistribution`.

    File format (SI units, ``%`` line comments)
    -------------------------------------------
    Columns: ``pos_x  pos_y  pos_z  mom_x  mom_y  mom_z  mass  charge  current``

    where ``mom_i`` is the normalised momentum :math:`\\beta_i \\gamma_i`
    (dimensionless), ``mass`` [kg] and ``charge`` [C, signed] are the
    per-particle physical species values, and ``current`` [A] is the
    macroparticle's outward emission current along its own velocity vector.

    A .pid file describes a **DC (steady-state) particle distribution**
    crossing a fixed plane, so :class:`SliceDistribution` is its natural
    container.  This reader assumes the slice plane is perpendicular to
    z (the standard beam-propagation direction); particles must lie on a
    common z-plane within ``plane_tol``.  A general 3D point cloud must be
    loaded as a :class:`ParticleDistribution3D` instead.

    The CST ``current`` is unprojected: converting it to the slice's linear
    charge density along z takes two steps — project onto the slice normal
    (``I_z = current * v_z / |v|``), then divide by ``v_z`` (``lam = I_z /
    v_z``).  The two factors collapse to ``lam = current / |v|``.

    Conversion
    ----------
    - x, y       : the two transverse position columns from the file.
    - z          : the common z₀ of the slice (mean of the file's z column
      when not given by the caller).
    - px, py, pz : ``mom * mass * c**2 / |charge|`` [eV/c], computed per row
      so heterogeneous species work.
    - t          : zeros (the format carries no time information).
    - lam        : ``current / |v|`` [C/m], with sign preserved.
    - extras     : raw ``current`` [A], ``mass`` [kg] and ``charge`` [C] are
      preserved as ``cst_current`` / ``cst_mass`` / ``cst_charge`` so no
      information is lost.

    Parameters
    ----------
    filepath
        Path to the .pid file.
    z
        z-position of the slice plane [m].  When ``None`` (default), the
        mean of the file's z column is used; when provided, every
        particle's z must match it to within ``plane_tol``.
    plane_tol
        Maximum allowed spread of the file's z column (and, when ``z`` is
        given, maximum allowed deviation from it), in meters.  Defaults to
        ``1e-9`` m.  Raises :class:`ValueError` if particles are not
        coplanar to this tolerance.
    dtype
        Data type passed to ``np.loadtxt``.
    """
    raw = _loadtxt_pid(filepath, dtype=dtype)

    pos_x, pos_y, pos_z = raw[:, 0], raw[:, 1], raw[:, 2]
    p_norm_x, p_norm_y, p_norm_z = raw[:, 3], raw[:, 4], raw[:, 5]
    mass = raw[:, 6]
    charge = raw[:, 7]
    current = raw[:, 8].astype(float)

    n = raw.shape[0]

    # Per-row rest energy m c^2 / |q|, used to convert mom (= beta*gamma) to eV/c.
    rest_energy_eV = mass * g_c**2 / np.abs(charge)
    px_eVc = p_norm_x * rest_energy_eV
    py_eVc = p_norm_y * rest_energy_eV
    pz_eVc = p_norm_z * rest_energy_eV

    # gamma from |p_norm|^2 = (beta*gamma)^2 = gamma^2 - 1.
    p_norm_sq = p_norm_x ** 2 + p_norm_y ** 2 + p_norm_z ** 2
    gamma = np.sqrt(1.0 + p_norm_sq)
    v_mag = np.sqrt(p_norm_sq) / gamma * g_c  # [m/s], total particle speed
    lam = current / v_mag                     # [C/m], sign-preserving

    spread = float(np.ptp(pos_z))
    if spread > plane_tol:
        raise ValueError(
            f"CST .pid particles are not coplanar along z: "
            f"position spread {spread:.3e} m exceeds plane_tol={plane_tol:.3e} m "
            f"(min={float(pos_z.min()):.6g}, max={float(pos_z.max()):.6g}). "
            f"Increase plane_tol, or load the file as a ParticleDistribution3D instead."
        )

    if z is None:
        z = float(np.mean(pos_z))
    else:
        deviation = float(np.max(np.abs(pos_z - z)))
        if deviation > plane_tol:
            raise ValueError(
                f"CST .pid particles deviate from z={z:.6g} m: max deviation "
                f"{deviation:.3e} m exceeds plane_tol={plane_tol:.3e} m."
            )

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
        z=z,
        x=pos_x,
        y=pos_y,
        px=px_eVc, py=py_eVc, pz=pz_eVc,
        t=np.zeros(n, dtype=float),
        lam=lam,
        extras=extras,
    )
