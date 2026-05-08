from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Union

import h5py
import numpy as np
from scipy.constants import c as g_c, m_e as g_m0, e as g_e0

from ..particle_array_quantity import ParticleArrayQuantity
from .core import ParticleDistribution


ASTRA_N_COLS = 10
# ASTRA_FMT = '%14.6E%14.6E%14.6E%14.6E%14.6E%14.6E%14.6E%14.6E%4d%4d'
ASTRA_FMT = '%12.4E'*8 + '%4d'*2  #exact original Astra format

@dataclass(frozen=True)
class AstraReferenceParticle:
    """
    ASTRA reference particle in internal units.

    Internal units
    --------------
    x, y, z    : m
    px, py, pz : eV/c
    t          : s
    Q          : C
    species    : int
    status     : int
    """
    x: float
    y: float
    z: float
    px: float
    py: float
    pz: float
    t: float
    Q: float
    species: int
    status: int

    @classmethod
    def from_raw_row(cls, row: np.ndarray) -> "AstraReferenceParticle":
        row = np.asarray(row, dtype=float).reshape(-1)
        if row.size < ASTRA_N_COLS:
            raise ValueError(
                f"ASTRA reference row must have at least {ASTRA_N_COLS} columns, got {row.size}."
            )

        return cls(
            x=float(row[0]),
            y=float(row[1]),
            z=float(row[2]),
            px=float(row[3]),
            py=float(row[4]),
            pz=float(row[5]),
            t=float(row[6]) * 1.0e-9,   # ns -> s
            Q=float(row[7]) * 1.0e-9,   # nC -> C
            species=int(round(row[8])),
            status=int(round(row[9])),
        )

    def to_raw_row(self) -> np.ndarray:
        """
        Convert to one ASTRA-format row.

        File units
        ----------
        x, y, z    : m
        px, py, pz : eV/c
        t          : ns
        Q          : nC
        """
        return np.array(
            [
                self.x,
                self.y,
                self.z,
                self.px,
                self.py,
                self.pz,
                self.t * 1.0e9,   # s -> ns
                self.Q * 1.0e9,   # C -> nC
                float(self.species),
                float(self.status),
            ],
            dtype=float,
        )


def _loadtxt_2d(filepath: str | Path, dtype: type = float) -> np.ndarray:
    arr = np.loadtxt(Path(filepath), dtype=dtype, ndmin=2)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {arr.shape}.")
    if arr.shape[1] < ASTRA_N_COLS:
        raise ValueError(
            f"ASTRA file must have at least {ASTRA_N_COLS} columns, got {arr.shape[1]}."
        )
    if arr.shape[0] < 1:
        raise ValueError("ASTRA file must contain at least one row.")
    return arr


def momentum_evc_to_velocity(
    px: np.ndarray,
    py: np.ndarray,
    pz: np.ndarray,
    *,
    m0: float = g_m0,
    e0: float = g_e0,
    c: float = g_c,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert momentum from [eV/c] to velocity [m/s].

    Parameters
    ----------
    px, py, pz
        Momentum components in [eV/c].
    m0
        Rest mass in kg. Default is electron rest mass.
    """
    px = np.asarray(px, dtype=float)
    py = np.asarray(py, dtype=float)
    pz = np.asarray(pz, dtype=float)

    factor = abs(e0) / c  # (eV/c) -> kg*m/s
    px_si = px * factor
    py_si = py * factor
    pz_si = pz * factor

    p2 = px_si**2 + py_si**2 + pz_si**2
    gamma = np.sqrt(1.0 + p2 / (m0 * c) ** 2)

    vx = px_si / (gamma * m0)
    vy = py_si / (gamma * m0)
    vz = pz_si / (gamma * m0)
    return vx, vy, vz


def velocity_to_momentum_evc(
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    *,
    m0: float = g_m0,
    e0: float = g_e0,
    c: float = g_c,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert velocity [m/s] to momentum [eV/c].

    Parameters
    ----------
    vx, vy, vz
        Velocity components in [m/s].
    m0
        Rest mass in kg. Default is electron rest mass.
    """
    vx = np.asarray(vx, dtype=float)
    vy = np.asarray(vy, dtype=float)
    vz = np.asarray(vz, dtype=float)

    v2 = vx**2 + vy**2 + vz**2
    beta2 = np.minimum(v2 / c**2, 1.0 - 1e-15)
    gamma = 1.0 / np.sqrt(1.0 - beta2)

    px_si = gamma * m0 * vx
    py_si = gamma * m0 * vy
    pz_si = gamma * m0 * vz

    factor = c / abs(e0)  # kg*m/s -> eV/c
    px = px_si * factor
    py = py_si * factor
    pz = pz_si * factor
    return px, py, pz


def _reference_to_distribution(
    ref: AstraReferenceParticle,
    *,
    species_key: str,
    status_key: str,
) -> ParticleDistribution:
    return ParticleDistribution(
        x=np.array([ref.x], dtype=float),
        y=np.array([ref.y], dtype=float),
        z=np.array([ref.z], dtype=float),
        px=np.array([ref.px], dtype=float),
        py=np.array([ref.py], dtype=float),
        pz=np.array([ref.pz], dtype=float),
        t=np.array([ref.t], dtype=float),
        Q=np.array([ref.Q], dtype=float),
        extras={
            species_key: ParticleArrayQuantity(
                name=species_key, data=np.array([ref.species], dtype=np.int64),
                unit="", dtype_kind="int", short_name=species_key,
                long_name="particle species flag", latex_name=species_key,
                category="flag", is_derived=False,
            ),
            status_key: ParticleArrayQuantity(
                name=status_key, data=np.array([ref.status], dtype=np.int64),
                unit="", dtype_kind="int", short_name=status_key,
                long_name="particle status flag", latex_name=status_key,
                category="flag", is_derived=False,
            ),
        },
    )


def _concat_distributions(
    first: ParticleDistribution,
    second: ParticleDistribution,
) -> ParticleDistribution:
    if set(first.extra_quantity_keys) != set(second.extra_quantity_keys):
        raise ValueError("Both ParticleDistribution objects must have identical extra keys.")

    extras: dict[str, ParticleArrayQuantity] = {}
    for key in first.extra_quantity_keys:
        q1 = first.get_quantity(key)
        q2 = second.get_quantity(key)
        extras[key] = ParticleArrayQuantity(
            name=key,
            data=np.concatenate([q1.data, q2.data]),
            unit=q1.unit,
            dtype_kind=q1.dtype_kind,
            short_name=q1.short_name,
            long_name=q1.long_name,
            latex_name=q1.latex_name,
            category=q1.category,
            is_derived=False,
            is_discrete=q1.is_discrete,
            preferred_scale=q1.preferred_scale,
        )

    return ParticleDistribution(
        x=np.concatenate([first.x, second.x]),
        y=np.concatenate([first.y, second.y]),
        z=np.concatenate([first.z, second.z]),
        px=np.concatenate([first.px, second.px]),
        py=np.concatenate([first.py, second.py]),
        pz=np.concatenate([first.pz, second.pz]),
        t=np.concatenate([first.t, second.t]),
        Q=np.concatenate([first.Q, second.Q]),
        extras=extras,
    )


def _build_reference_from_distribution(
    dist: ParticleDistribution,
    *,
    mode: str = "mean",
    weight: str | np.ndarray | None = "Q_abs",
    reference_time: float = 0.0,
    reference_charge: float = 0.0,
    default_species: int = 1,
    default_status: int = 5,
    species_key: str = "species",
    status_key: str = "status",
) -> AstraReferenceParticle:
    """
    Build an ASTRA reference particle from ParticleDistribution.

    mode
    ----
    - "mean"  : use weighted means for x, y, z, px, py, pz
    - "zeros" : use zeros for x, y, z, px, py, pz
    """
    if mode == "mean":
        x0  = dist.mean("x",  weight=weight)
        y0  = dist.mean("y",  weight=weight)
        z0  = dist.mean("z",  weight=weight)
        px0 = dist.mean("px", weight=weight)
        py0 = dist.mean("py", weight=weight)
        pz0 = dist.mean("pz", weight=weight)
    elif mode == "zeros":
        x0 = y0 = z0 = px0 = py0 = pz0 = 0.0
    else:
        raise ValueError("mode must be 'mean' or 'zeros'.")

    return AstraReferenceParticle(
        x=float(x0),
        y=float(y0),
        z=float(z0),
        px=float(px0),
        py=float(py0),
        pz=float(pz0),
        t=float(reference_time),
        Q=float(reference_charge),
        species=int(default_species),
        status=int(default_status),
    )


def read_astra_distribution(
    filepath: str | Path,
    *,
    include_reference_particle: bool = True,
    return_reference: bool = False,
    species_key: str = "species",
    status_key: str = "status",
    dtype: type = float,
) -> ParticleDistribution | tuple[ParticleDistribution, AstraReferenceParticle]:
    """
    Read an ASTRA distribution file into ParticleDistribution.

    ASTRA convention used here
    --------------------------
    - Row 0 is the reference particle in absolute values.
    - Rows 1: store x, y, z, px, py, pz, t relative to the reference particle.
    - Q is absolute.
    - Columns 9 and 10 are integer flags:
      * species
      * status

    Returned ParticleDistribution convention
    ----------------------------------------
    - Quantities are converted to absolute physical values.
    - Units in memory are:
        x, y, z   : m
        px, py, pz: eV/c
        t         : s
        Q         : C

    Parameters
    ----------
    include_reference_particle
        If True, include the ASTRA reference particle itself as the first particle
        in the returned ParticleDistribution. Default is True.
    return_reference
        If True, also return the parsed AstraReferenceParticle.
    """
    raw = _loadtxt_2d(filepath, dtype=dtype)
    ref = AstraReferenceParticle.from_raw_row(raw[0])

    data = raw[1:, :]
    n = data.shape[0]

    if n == 0:
        real_dist = ParticleDistribution(
            x=np.empty(0, dtype=float),
            y=np.empty(0, dtype=float),
            z=np.empty(0, dtype=float),
            px=np.empty(0, dtype=float),
            py=np.empty(0, dtype=float),
            pz=np.empty(0, dtype=float),
            t=np.empty(0, dtype=float),
            Q=np.empty(0, dtype=float),
            extras={
                species_key: ParticleArrayQuantity(
                    name=species_key,
                    data=np.empty(0, dtype=np.int64),
                    unit="",
                    dtype_kind="int",
                    short_name=species_key,
                    long_name="particle species flag",
                    latex_name=species_key,
                    category="flag",
                    is_derived=False,
                ),
                status_key: ParticleArrayQuantity(
                    name=status_key,
                    data=np.empty(0, dtype=np.int64),
                    unit="",
                    dtype_kind="int",
                    short_name=status_key,
                    long_name="particle status flag",
                    latex_name=status_key,
                    category="flag",
                    is_derived=False,
                ),
            },
        )
    else:
        x  = data[:, 0] + ref.x
        y  = data[:, 1] + ref.y
        z  = data[:, 2] + ref.z

        px = data[:, 3] + ref.px
        py = data[:, 4] + ref.py
        pz = data[:, 5] + ref.pz

        t  = data[:, 6] * 1.0e-9 + ref.t   # ns -> s, then add reference time
        Q  = data[:, 7] * 1.0e-9            # nC -> C, absolute

        species = np.rint(data[:, 8]).astype(np.int64)
        status  = np.rint(data[:, 9]).astype(np.int64)

        real_dist = ParticleDistribution(
            x=x, y=y, z=z,
            px=px, py=py, pz=pz,
            t=t, Q=Q,
            extras={
                species_key: ParticleArrayQuantity(
                    name=species_key, data=species, unit="", dtype_kind="int",
                    short_name=species_key, long_name="particle species flag",
                    latex_name=species_key, category="flag", is_derived=False,
                ),
                status_key: ParticleArrayQuantity(
                    name=status_key, data=status, unit="", dtype_kind="int",
                    short_name=status_key, long_name="particle status flag",
                    latex_name=status_key, category="flag", is_derived=False,
                ),
            },
        )

    if include_reference_particle:
        ref_dist = _reference_to_distribution(
            ref,
            species_key=species_key,
            status_key=status_key,
        )
        dist = _concat_distributions(ref_dist, real_dist)
    else:
        dist = real_dist

    return (dist, ref) if return_reference else dist


def write_astra_distribution(
    filepath: str | Path,
    dist: ParticleDistribution,
    *,
    reference_particle: AstraReferenceParticle | None = None,
    include_reference_particle: bool = True,
    reference_mode: str = "keep",
    species_key: str = "species",
    status_key: str = "status",
    default_species: int = 1,
    default_status: int = 5,
    weight: str | np.ndarray | None = "Q_abs",
    fmt: str = ASTRA_FMT,
    delimiter: str = " ",
) -> None:
    """
    Write ParticleDistribution to an ASTRA distribution file.

    ASTRA convention written here
    -----------------------------
    - Row 0 is the reference particle in absolute values.
    - Rows 1: store x, y, z, px, py, pz, t relative to row 0.
    - Q is written as absolute macro charge in nC.
    - species/status are written from extras as integer-like columns.

    Parameters
    ----------
    reference_particle
        If given, use it directly as row 0.
    include_reference_particle
        If True and reference_particle is None, treat dist[0] as the ASTRA
        reference particle. Default is True.
    reference_mode
        Used only when reference_particle is None and include_reference_particle=False.
        - "keep"  : error
        - "mean"  : build from weighted centroid
        - "zeros" : build zero reference
    """
    n_total = dist.size

    if reference_particle is not None:
        ref = reference_particle
        dist_particles = dist
    else:
        if include_reference_particle:
            if n_total < 1:
                raise ValueError(
                    "dist must contain at least one particle when include_reference_particle=True."
                )

            if species_key in dist.extra_quantity_keys:
                species0 = int(dist.get_data(species_key)[0])
            else:
                species0 = int(default_species)

            if status_key in dist.extra_quantity_keys:
                status0 = int(dist.get_data(status_key)[0])
            else:
                status0 = int(default_status)

            ref = AstraReferenceParticle(
                x=float(dist.x[0]),   y=float(dist.y[0]),   z=float(dist.z[0]),
                px=float(dist.px[0]), py=float(dist.py[0]), pz=float(dist.pz[0]),
                t=float(dist.t[0]),   Q=float(dist.Q[0]),
                species=species0,      status=status0,
            )
            dist_particles = dist.slice(slice(1, None))
        else:
            if reference_mode == "keep":
                raise ValueError(
                    "reference_particle must be provided when "
                    "reference_mode='keep' and include_reference_particle=False."
                )
            ref = _build_reference_from_distribution(
                dist,
                mode=reference_mode,
                weight=weight,
                default_species=default_species,
                default_status=default_status,
                species_key=species_key,
                status_key=status_key,
            )
            dist_particles = dist

    n = dist_particles.size

    if species_key in dist_particles.extra_quantity_keys:
        species_arr = np.asarray(dist_particles.get_data(species_key), dtype=np.int64)
    else:
        species_arr = np.full(n, default_species, dtype=np.int64)

    if status_key in dist_particles.extra_quantity_keys:
        status_arr = np.asarray(dist_particles.get_data(status_key), dtype=np.int64)
    else:
        status_arr = np.full(n, default_status, dtype=np.int64)

    if species_arr.shape != (n,):
        raise ValueError(
            f"extra[{species_key!r}] must have shape ({n},), got {species_arr.shape}."
        )
    if status_arr.shape != (n,):
        raise ValueError(
            f"extra[{status_key!r}] must have shape ({n},), got {status_arr.shape}."
        )

    px = dist_particles.px
    py = dist_particles.py
    pz = dist_particles.pz

    raw = np.empty((n + 1, ASTRA_N_COLS), dtype=float)
    raw[0, :] = ref.to_raw_row()

    raw[1:, 0] = np.asarray(dist_particles.x, dtype=float) - ref.x
    raw[1:, 1] = np.asarray(dist_particles.y, dtype=float) - ref.y
    raw[1:, 2] = np.asarray(dist_particles.z, dtype=float) - ref.z

    raw[1:, 3] = px - ref.px
    raw[1:, 4] = py - ref.py
    raw[1:, 5] = pz - ref.pz

    raw[1:, 6] = (np.asarray(dist_particles.t, dtype=float) - ref.t) * 1.0e9
    raw[1:, 7] = np.asarray(dist_particles.Q, dtype=float) * 1.0e9

    raw[1:, 8] = species_arr.astype(float)
    raw[1:, 9] = status_arr.astype(float)

    np.savetxt(
        Path(filepath),
        raw,
        fmt=fmt,
        delimiter=delimiter,
    )


def from_ocelot_particle_array(pa: "Any") -> ParticleDistribution:
    """
    Convert an ocelot ``ParticleArray`` to a :class:`ParticleDistribution`.

    ocelot ``ParticleArray`` coordinate layout
    -------------------------------------------
    ``rparticles[0]``: *x*   — horizontal position [m]
    ``rparticles[1]``: *x'*  = px/p₀ — horizontal divergence [rad]
    ``rparticles[2]``: *y*   — vertical position [m]
    ``rparticles[3]``: *y'*  = py/p₀ — vertical divergence [rad]
    ``rparticles[4]``: *tau* — longitudinal position offset [m],
                       defined as ``tau = pa.s − z``.
                       Positive tau → particle is *behind* the reference
                       (smaller z, arrives later).
    ``rparticles[5]``: *p*   = (E − E₀) / (p₀c) — relative energy deviation
    ``q_array``       : macro-particle charge [C]
    ``E``             : reference energy [GeV]
    ``s``             : reference position along the beamline [m]

    Note on tau
    -----------
    Despite the name, ocelot's tau coordinate is **not** time; it is a
    longitudinal *position* offset in metres (``tau = pa.s − z``).
    This is confirmed by:

    * ``std(tau) * 1e3`` is printed in "mm" in ocelot's ``__repr__``.
    * ``t = parray.tau() / speed_of_light`` (ocelot source) divides tau by c
      to obtain time, so tau must be in m.
    * ``z = p_array.tau()`` is used directly as a position in ocelot utilities.

    Returned :class:`ParticleDistribution` conventions
    ---------------------------------------------------
    ``x``, ``y``    : transverse positions [m]  (unchanged)
    ``z``           : ``pa.s − tau``  [m]
    ``px``, ``py``  : transverse momenta [eV/c]  = x′/y′ × p₀c
    ``pz``          : longitudinal momentum [eV/c]  (derived from total energy)
    ``t``           : ``z / c``  [s]  (ultra-relativistic approximation, β ≈ 1)
    ``Q``           : ``pa.q_array``  [C]
    """
    m_e_eV: float = g_m0 * g_c ** 2 / abs(g_e0)   # electron rest energy [eV]

    E0_GeV: float = float(pa.E)
    p0c_GeV: float = float(pa.p0c)       # sqrt(E² − mₑ²) in GeV
    p0c_eV: float = p0c_GeV * 1e9        # [eV]

    x   = np.array(pa.rparticles[0], dtype=float)
    xp  = np.array(pa.rparticles[1], dtype=float)  # x' = px / p₀
    y   = np.array(pa.rparticles[2], dtype=float)
    yp  = np.array(pa.rparticles[3], dtype=float)  # y' = py / p₀
    tau = np.array(pa.rparticles[4], dtype=float)  # longitudinal position offset [m]
    dp  = np.array(pa.rparticles[5], dtype=float)  # (E − E₀) / p₀c  (dimensionless)

    # Transverse momenta [eV/c]
    px_evc = xp * p0c_eV
    py_evc = yp * p0c_eV

    # Total energy and longitudinal momentum
    E_eV = (dp * p0c_GeV + E0_GeV) * 1e9           # [eV]
    p_abs_sq = np.maximum(E_eV ** 2 - m_e_eV ** 2, 0.0)
    pz_evc = np.sqrt(np.maximum(p_abs_sq - px_evc ** 2 - py_evc ** 2, 0.0))

    # Longitudinal position: tau = pa.s − z  →  z = pa.s − tau
    z = float(pa.s) - tau          # [m]

    # Time: ultra-relativistic approximation (β ≈ 1)
    t = z / g_c                    # [s]

    return ParticleDistribution(
        x=x, y=y, z=z,
        px=px_evc, py=py_evc, pz=pz_evc,
        t=t, Q=np.array(pa.q_array, dtype=float),
    )


def to_ocelot_particle_array(
    dist: ParticleDistribution,
    *,
    s: float = 0.0,
) -> "Any":
    """
    Convert a :class:`ParticleDistribution` to an ocelot ``ParticleArray``.

    The *reference particle* is defined as the charge-weighted centroid of the
    distribution.  Its reference position along the beamline is ``s``
    (default 0).

    ocelot ``ParticleArray`` conventions used
    -----------------------------------------
    ``rparticles[0]``: *x*   [m]  (unchanged)
    ``rparticles[1]``: *x'*  = px / p₀c  [rad]
    ``rparticles[2]``: *y*   [m]  (unchanged)
    ``rparticles[3]``: *y'*  = py / p₀c  [rad]
    ``rparticles[4]``: *tau* = s − z  [m]  (positive → behind reference)
    ``rparticles[5]``: *p*   = (E − E₀) / p₀c  (dimensionless)
    ``q_array``       : macro-particle charge [C]
    ``E``             : charge-weighted mean energy [GeV]
    ``s``             : ``s`` parameter [m]

    Parameters
    ----------
    dist
        Source particle distribution.
    s
        Reference position along the beamline [m].  Typically set to the
        longitudinal position at which the distribution was recorded.
        Default is 0.
    """
    from ocelot.cpbd.beam import ParticleArray

    m_e_eV: float = g_m0 * g_c ** 2 / abs(g_e0)

    px_evc = np.asarray(dist.px, dtype=float)
    py_evc = np.asarray(dist.py, dtype=float)
    pz_evc = np.asarray(dist.pz, dtype=float)

    # Total energy per particle [eV]
    E_eV = np.sqrt(px_evc ** 2 + py_evc ** 2 + pz_evc ** 2 + m_e_eV ** 2)

    # Charge-weighted reference energy
    weights = np.abs(np.asarray(dist.Q, dtype=float))
    total_weight = float(weights.sum())
    if total_weight > 0.0:
        E0_eV = float(np.dot(weights, E_eV) / total_weight)
    else:
        E0_eV = float(np.mean(E_eV))

    E0_GeV = E0_eV * 1e-9
    p0c_eV = float(np.sqrt(max(E0_eV ** 2 - m_e_eV ** 2, 0.0)))

    # ocelot phase-space coordinates
    xp    = px_evc / p0c_eV                                        # x' [rad]
    yp    = py_evc / p0c_eV                                        # y' [rad]
    tau   = float(s) - np.asarray(dist.z, dtype=float)             # tau = s − z  [m]
    delta = (E_eV - E0_eV) / p0c_eV                               # p = (E−E₀)/p₀c

    n = dist.size
    pa = ParticleArray(n=n)
    pa.rparticles[0] = np.asarray(dist.x, dtype=float)
    pa.rparticles[1] = xp
    pa.rparticles[2] = np.asarray(dist.y, dtype=float)
    pa.rparticles[3] = yp
    pa.rparticles[4] = tau
    pa.rparticles[5] = delta
    pa.q_array[:] = np.asarray(dist.Q, dtype=float)
    pa.E = E0_GeV
    pa.s = float(s)

    return pa


def read_genesis_distribution(
    filepath: str | Path,
    *,
    drop_zero_charge: bool = True,
) -> ParticleDistribution:
    """
    Read a GENESIS .h5 particle distribution file into ParticleDistribution.

    GENESIS .h5 file format (as observed in scan.000.out.par.h5):
    - Root contains global datasets: slicelength, slicespacing, refposition,
      beamletsize, one4one, slicecount.
    - Slice groups named slice000001, slice000002, ... each containing:
        current: scalar (A)
        x, y: particle positions (m)
        gamma: Lorentz factor
        px, py: dimensionless momentum components (p/(m_e c))
        theta: phase (rad)

    The longitudinal position z is computed from theta as:
        z_relative = (theta / (2π) + 0.5) * slicelength
        z = z_relative + refposition + i * slicespacing
    where i is the slice index (0‑based).

    The charge per particle in a slice is computed as:
        Q_slice = current * slicelength / c
        Q_per_particle = Q_slice / n_particles_in_slice

    Particles from slices with zero current (hence zero charge) are discarded
    when drop_zero_charge=True (default).

    Parameters
    ----------
    filepath
        Path to the .h5 file.
    drop_zero_charge
        If True, discard particles with zero macro charge (from slices with
        zero current). Default is True.

    Returns
    -------
    ParticleDistribution
        With units:
            x, y, z: m
            px, py, pz: eV/c
            t: s
            Q: C
        No extra quantities are added.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    with h5py.File(filepath, 'r') as f:
        # Global parameters
        slicelength = float(f['slicelength'][()].item())  # m
        slicespacing = float(f['slicespacing'][()].item())  # m
        refposition = float(f['refposition'][()].item())  # m
        # beamletsize and one4one are not used here

        # Speed of light
        c = g_c  # m/s

        # List slice groups (exclude datasets like slicecount, slicelength, slicespacing)
        slice_keys = [k for k in f.keys() if k.startswith('slice') and isinstance(f[k], h5py.Group)]
        slice_keys.sort()  # ensure order

        if not slice_keys:
            raise ValueError(f"No slice groups found in {filepath}")

        if 'slicecount' in f:
            expected_slices = int(f['slicecount'][()].item())
            if len(slice_keys) != expected_slices:
                raise ValueError(
                    f"Expected {expected_slices} slice groups (from 'slicecount'), "
                    f"found {len(slice_keys)} in {filepath}"
                )

        # Particles per slice (assumed constant)
        first_slice = f[slice_keys[0]]
        n_per_slice = first_slice['x'].shape[0]

        # Pre‑allocate lists
        all_x = []
        all_y = []
        all_z = []
        all_px = []
        all_py = []
        all_pz = []
        all_t = []
        all_Q = []

        for i, sk in enumerate(slice_keys):
            grp = f[sk]
            # Current is stored as a 1‑element array
            current = float(grp['current'][()].item())  # A

            # Charge per particle in this slice
            Q_slice = current * slicelength / c  # C
            Q_per_particle = Q_slice / n_per_slice  # C

            if drop_zero_charge and Q_per_particle == 0:
                continue

            # Particle data
            x = grp['x'][:]  # m
            y = grp['y'][:]  # m
            gamma = grp['gamma'][:]  # dimensionless
            px_norm = grp['px'][:]  # dimensionless p_x/(m_e c)
            py_norm = grp['py'][:]  # dimensionless p_y/(m_e c)
            theta = grp['theta'][:]  # rad

            n_this = len(x)
            if n_this != n_per_slice:
                # Recompute charge per particle based on actual count
                Q_per_particle = Q_slice / n_this if n_this > 0 else 0.0

            # Compute pz from gamma and px, py (all in units of m_e c).
            # gamma^2 = 1 + px^2 + py^2 + pz^2
            # Clamp to zero before sqrt to absorb floating‑point rounding errors.
            # pz is taken as positive (forward-propagating beam assumed).
            pz_squared = gamma**2 - 1 - px_norm**2 - py_norm**2
            pz_squared = np.maximum(pz_squared, 0.0)
            pz_norm = np.sqrt(pz_squared)

            # Convert dimensionless momentum to eV/c
            # factor_evc = m_e * c^2 / e0 ≈ 510998.946 eV
            factor_evc = g_m0 * g_c**2 / abs(g_e0)
            px_evc = px_norm * factor_evc
            py_evc = py_norm * factor_evc
            pz_evc = pz_norm * factor_evc

            # Longitudinal position from theta
            # theta ∈ [0, 2π) maps to z_relative ∈ [0, slicelength)
            z_relative = theta / (2 * np.pi) * slicelength  # m
            z = z_relative + refposition + i * slicespacing  # m

            # Time: t ≈ z / c (valid for ultra-relativistic particles, β ≈ 1)
            t = z / c  # s

            # Charge per particle
            Q = np.full(n_this, Q_per_particle, dtype=float)

            # Append to lists
            all_x.append(x)
            all_y.append(y)
            all_z.append(z)
            all_px.append(px_evc)
            all_py.append(py_evc)
            all_pz.append(pz_evc)
            all_t.append(t)
            all_Q.append(Q)

        if not all_x:
            # No particles selected
            empty = np.empty(0, dtype=float)
            return ParticleDistribution(
                x=empty, y=empty, z=empty,
                px=empty, py=empty, pz=empty,
                t=empty, Q=empty,
                extras={},
            )

        # Concatenate
        x_arr = np.concatenate(all_x)
        y_arr = np.concatenate(all_y)
        z_arr = np.concatenate(all_z)
        px_arr = np.concatenate(all_px)
        py_arr = np.concatenate(all_py)
        pz_arr = np.concatenate(all_pz)
        t_arr = np.concatenate(all_t)
        Q_arr = np.concatenate(all_Q)

    return ParticleDistribution(
        x=x_arr, y=y_arr, z=z_arr,
        px=px_arr, py=py_arr, pz=pz_arr,
        t=t_arr, Q=Q_arr,
        extras={},
    )


# ---------------------------------------------------------------------------
# Genesis write helpers (private)
# ---------------------------------------------------------------------------

def _warn_or_raise(msg: str, mode: str) -> None:
    if mode == "raise":
        raise ValueError(msg)
    elif mode == "warn":
        warnings.warn(msg, UserWarning, stacklevel=4)


def _resample_slice_particles(
    arr6d: np.ndarray,
    npart: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Resample a (N, 6) particle array [x, y, theta, px, py, gamma] to exactly
    *npart* rows using the Genesis nearest-neighbour interpolation algorithm.

    - If N >= npart: randomly delete rows down to npart.
    - If N < npart: keep all N rows, add npart-N synthetic rows by interpolating
      between each selected particle and its pre-computed nearest neighbour in
      normalised 6-D space.  The theta column (index 2) is folded back to
      [0, 2π) after interpolation.

    Caller must ensure N >= 2.
    """
    arr = np.array(arr6d, dtype=float)
    n = len(arr)

    if n >= npart:
        # Randomly thin down to npart
        idx = rng.choice(n, size=npart, replace=False)
        return arr[idx]

    out = np.empty((npart, 6), dtype=float)
    out[:n] = arr

    # Normalise to zero-mean, unit-RMS in each dimension
    avg = arr.mean(axis=0)
    std = arr.std(axis=0)
    std_inv = np.where(std == 0.0, 1.0, 1.0 / std)
    arr_norm = (arr - avg) * std_inv

    # Sort by theta (column 2) to localise the nearest-neighbour window
    sort_idx = np.argsort(arr_norm[:, 2])
    arr_norm = arr_norm[sort_idx]

    # Pre-compute nearest neighbour for each of the N input particles
    hw = min(1024, n - 1)
    closest = np.empty(n, dtype=int)
    for i in range(n):
        lo = max(0, i - hw)
        hi = min(n, i + hw)
        diff = arr_norm[lo:hi] - arr_norm[i]
        dist2 = np.einsum("ij,ij->i", diff, diff)
        dist2[i - lo] = np.inf  # exclude self
        closest[i] = lo + int(np.argmin(dist2))

    # Add npart - n synthetic particles
    for k in range(n, npart):
        n1 = int(rng.integers(n))
        n2 = int(closest[n1])
        r = float(rng.random()) * 2.0 - 1.0  # uniform ∈ (-1, 1)
        new_norm = 0.5 * (arr_norm[n1] + arr_norm[n2]) + r * (arr_norm[n1] - arr_norm[n2])
        out[k] = new_norm / std_inv + avg

    # Fold theta back to [0, 2π)
    out[:, 2] = out[:, 2] % (2.0 * np.pi)
    return out


def _build_longitudinal_cdf(
    z: np.ndarray,
    q: np.ndarray,
    *,
    bin_width: float,
    savgol_window: int = 17,
    savgol_degree: int = 4,
):
    """
    Build a smoothed cumulative distribution function of the longitudinal
    charge profile.

    Returns ``(Fx, Fx_inv)`` where:
      - ``Fx(z)``   -> cumulative charge fraction in [0, 1]
      - ``Fx_inv(u)`` -> z position for cumulative fraction u

    Returns ``None`` if the distribution is empty or degenerate.
    """
    from scipy.interpolate import interp1d
    from scipy.signal import savgol_filter

    if len(z) < 2:
        return None
    z_min, z_max = float(z.min()), float(z.max())
    if z_max <= z_min:
        return None

    n_bins = max(int((z_max - z_min) / bin_width), 4)
    charge_hist, edges = np.histogram(z, bins=n_bins, weights=q)
    ds = float(edges[1] - edges[0])

    # Pad with zeros on each side to reduce boundary artefacts
    pad = 9
    ch_pad = np.concatenate(([0.0] * pad, charge_hist.astype(float), [0.0] * pad))

    # Savitzky-Golay smoothing
    win = min(savgol_window, len(ch_pad))
    win = win if win % 2 == 1 else win - 1
    win = max(win, 3)
    poly = min(savgol_degree, win - 1)
    ch_pad = savgol_filter(ch_pad, win, poly)
    ch_pad = np.maximum(ch_pad, 0.0)

    # Trim padding back to n_bins values; bin centres from original edges
    ch_smooth = ch_pad[pad: pad + n_bins]
    bin_centers = 0.5 * (edges[:-1] + edges[1:])

    total = float(ch_smooth.sum())
    if total <= 0.0:
        return None
    ch_smooth /= total

    # Build CDF: prepend a zero at (first_center - ds) so CDF starts at 0
    z_cdf = np.concatenate(([bin_centers[0] - ds], bin_centers))
    cdf = np.concatenate(([0.0], np.cumsum(ch_smooth)))
    cdf[-1] = 1.0  # ensure exact 1

    Fx = interp1d(z_cdf, cdf, kind="linear", bounds_error=False,
                  fill_value=(0.0, 1.0))

    # Deduplicate CDF values (can appear when smoothed bins are zero)
    _, uid = np.unique(cdf, return_index=True)
    Fx_inv = interp1d(cdf[uid], z_cdf[uid], kind="linear", bounds_error=False,
                      fill_value=(float(z_cdf[0]), float(z_cdf[-1])))

    return Fx, Fx_inv


def _make_zero_current_slice_from_template(template: dict[str, np.ndarray]) -> dict:
    """
    Return a zero-current slice using a non-degenerate transverse template.

    The copied particles are format fillers only; the slice current remains
    zero so Genesis should treat the slice as empty in current-weighted
    physics while still avoiding singular all-particles-overlap geometry.
    """
    npart = int(np.asarray(template["x"]).size)
    return dict(
        x=np.asarray(template["x"], dtype=float).copy(),
        y=np.asarray(template["y"], dtype=float).copy(),
        theta=np.linspace(0.0, 2.0 * np.pi, npart, endpoint=False),
        px=np.asarray(template["px"], dtype=float).copy(),
        py=np.asarray(template["py"], dtype=float).copy(),
        gamma=np.asarray(template["gamma"], dtype=float).copy(),
        current=0.0,
    )


def _write_dataset_with_unit(
    group: h5py.Group | h5py.File,
    name: str,
    data: Any,
    unit: str | None = None,
):
    ds = group.create_dataset(name, data=data)
    if unit is not None:
        ds.attrs["unit"] = unit
    return ds


# ---------------------------------------------------------------------------
# Genesis write (public)
# ---------------------------------------------------------------------------

_TWO_PI = 2.0 * np.pi
_ME_EV = g_m0 * g_c**2 / abs(g_e0)  # m_e c^2 in eV  ≈ 510998.95 eV


def write_genesis_distribution(
    dist: "ParticleDistribution",
    filepath: Union[str, Path],
    *,
    lambda0: float,
    s0: float,
    slen: float,
    npart: int = 4096,
    sample: int = 1,
    mpi_nproc: int = 1,
    z0: Union[float, str, None] = None,
    theta_method: Literal["direct", "hammersley", "penman"] = "direct",
    smooth_current: bool = False,
    min_particles: int = 2,
    on_warning: Literal["silent", "warn", "raise"] = "warn",
    version: int = 4,
    nbins: int = 16,
    seed: Optional[int] = None,
) -> None:
    """
    Write a GENESIS 1.3 (v3/v4) HDF5 particle distribution file.

    The beam is divided into time slices of width ``lambda0`` (the radiation
    wavelength) spaced ``lambda0 * sample`` apart.  Each slice contains
    exactly ``npart`` macro-particles described by
    ``x, y, theta, px, py, gamma``.

    Parameters
    ----------
    dist
        Input particle distribution.
    filepath
        Output ``.h5`` file path.
    lambda0
        Radiation wavelength [m].
    s0
        Start of the time window [m].
    slen
        Length of the time window [m].
    npart
        Number of macro-particles per slice.  Default 4096.
    sample
        Slice spacing in units of ``lambda0`` (positive integer).
        ``sample=1`` gives one slice per wavelength (densest).
        Default 1.
    mpi_nproc
        Expected number of MPI processes.  The total slice count is rounded
        up to the nearest multiple of ``mpi_nproc``.  Default 1.
    z0
        Longitudinal shift applied to the distribution before slicing.

        - ``None``: no shift (default).
        - ``float``: shift the charge-weighted centroid to this position.
        - ``"auto"``: shift the leftmost particle to ``s0``.
    theta_method
        How particle phases (theta) within each slice are assigned.

        - ``"direct"`` (default): theta computed directly from each
          particle's z position within the slice.
        - ``"hammersley"``: theta redistribution using a Halton
          quasi-random sequence mapped through the smooth CDF.
          Minimises initial shot noise; recommended for SASE FEL.
        - ``"penman"``: same as ``"hammersley"`` but uses the Penman
          (1992) regular-grid-plus-jitter method instead.

        For ``"hammersley"`` and ``"penman"``, the smooth CDF is always
        built regardless of ``smooth_current``.
    smooth_current
        If ``True``, compute slice currents from the Savitzky-Golay-
        smoothed CDF instead of a direct particle count.  Automatically
        ``True`` when ``theta_method != "direct"``.  Default ``False``.
    min_particles
        Minimum number of particles required in a slice to attempt
        resampling.  Slices below this threshold are treated as
        zero-current.  Default 2.
    on_warning
        How to handle recoverable issues (nslice adjustment, out-of-window
        particles).

        - ``"warn"`` (default): emit a ``UserWarning`` and proceed.
        - ``"silent"``: proceed silently.
        - ``"raise"``: raise ``ValueError``.
    version
        Genesis version (3 or 4).  Affects ``one4one`` flag and version
        metadata written to the file.  Default 4.
    nbins
        Beamlet size written to ``beamletsize`` in the output file.
        Default 16 for backward interface compatibility.
    seed
        Random seed.  If ``None``, a random seed is chosen.
    """
    # ------------------------------------------------------------------
    # 1. Parameter validation
    # ------------------------------------------------------------------
    if not isinstance(sample, int) or sample < 1:
        raise ValueError(f"sample must be a positive integer, got {sample!r}.")
    if not isinstance(mpi_nproc, int) or mpi_nproc < 1:
        raise ValueError(f"mpi_nproc must be a positive integer, got {mpi_nproc!r}.")
    if version not in (3, 4):
        raise ValueError(f"version must be 3 or 4, got {version!r}.")
    if not isinstance(nbins, int) or nbins < 1:
        raise ValueError(f"nbins must be a positive integer, got {nbins!r}.")
    if theta_method not in ("direct", "hammersley", "penman"):
        raise ValueError(
            f"theta_method must be 'direct', 'hammersley', or 'penman', "
            f"got {theta_method!r}."
        )
    if on_warning not in ("silent", "warn", "raise"):
        raise ValueError(
            f"on_warning must be 'silent', 'warn', or 'raise', "
            f"got {on_warning!r}."
        )

    slicespacing = lambda0 * sample
    slicelength = lambda0

    # ------------------------------------------------------------------
    # 2. Compute nslice
    # ------------------------------------------------------------------
    import math
    n0 = math.ceil(slen / slicespacing)
    if n0 % mpi_nproc != 0:
        n0_adj = math.ceil(n0 / mpi_nproc) * mpi_nproc
        _warn_or_raise(
            f"nslice adjusted from {n0} to {n0_adj} to be a multiple of "
            f"mpi_nproc={mpi_nproc}.",
            on_warning,
        )
        n0 = n0_adj
    nslice = n0

    # ------------------------------------------------------------------
    # 3. Extract arrays
    # ------------------------------------------------------------------
    z = dist.get_data("z").copy().astype(float)
    x = dist.get_data("x").astype(float)
    y = dist.get_data("y").astype(float)
    px_evc = dist.get_data("px").astype(float)
    py_evc = dist.get_data("py").astype(float)
    pz_evc = dist.get_data("pz").astype(float)
    Q_abs = np.abs(dist.get_data("Q").astype(float))

    px_norm = px_evc / _ME_EV
    py_norm = py_evc / _ME_EV
    p_abs_norm = np.sqrt(px_norm**2 + py_norm**2 + (pz_evc / _ME_EV)**2)
    gamma = np.sqrt(1.0 + p_abs_norm**2)

    Qb = float(Q_abs.sum())
    w = Q_abs
    # ------------------------------------------------------------------
    # 4. RNG and Halton setup
    # ------------------------------------------------------------------
    if seed is None:
        seed = int(np.random.randint(0, 2**31))
    rng = np.random.default_rng(seed)

    global_template = None
    if len(z) >= min_particles:
        theta_global = ((z - s0) / slicelength * _TWO_PI) % _TWO_PI
        arr6d_global = np.column_stack([x, y, theta_global, px_norm, py_norm, gamma])
        resampled_global = _resample_slice_particles(arr6d_global, npart, rng)
        global_template = dict(
            x=resampled_global[:, 0],
            y=resampled_global[:, 1],
            px=resampled_global[:, 3],
            py=resampled_global[:, 4],
            gamma=resampled_global[:, 5],
        )

    # ------------------------------------------------------------------
    # 5. Apply z0 offset
    # ------------------------------------------------------------------
    if z0 is not None:
        if isinstance(z0, str):
            if z0 == "auto":
                z_shift = s0 - float(z.min())
            else:
                raise ValueError(f"z0 string must be 'auto', got {z0!r}.")
        else:
            z_mean = float(np.average(z, weights=w))
            z_shift = float(z0) - z_mean
        z = z + z_shift

    # ------------------------------------------------------------------
    # 6. Check window bounds
    # ------------------------------------------------------------------
    s1 = s0 + slen
    mask_out = (z < s0) | (z >= s1)
    n_out = int(mask_out.sum())
    if n_out > 0:
        _warn_or_raise(
            f"{n_out} particles are outside the window [s0, s0+slen] and "
            f"will not appear in any slice.",
            on_warning,
        )

    # ------------------------------------------------------------------
    # 7. Build CDF if needed
    # ------------------------------------------------------------------
    need_cdf = smooth_current or theta_method in ("hammersley", "penman")
    cdf_result = None
    if need_cdf:
        mask_in = ~mask_out
        cdf_result = _build_longitudinal_cdf(
            z[mask_in], Q_abs[mask_in],
            bin_width=lambda0 / 10.0,
        )
        if cdf_result is None:
            _warn_or_raise(
                "Could not build longitudinal CDF (too few particles or "
                "degenerate distribution). Falling back to direct current "
                "and theta computation.",
                on_warning,
            )
            need_cdf = False
            smooth_current = False
        else:
            Fx, Fx_inv = cdf_result

    halton_samples: Optional[np.ndarray] = None
    halton_idx = 0
    if theta_method == "hammersley":
        from scipy.stats.qmc import Halton
        halton = Halton(d=1, scramble=True, seed=seed)
        halton_samples = halton.random(nslice * npart).ravel()

    # ------------------------------------------------------------------
    # 8. Write HDF5 file
    # ------------------------------------------------------------------
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(filepath, "w") as f:
        template_slice = global_template
        for islice in range(nslice):
            slice_left = s0 + islice * slicespacing
            slice_right = slice_left + slicelength

            mask_slice = (z >= slice_left) & (z < slice_right)
            n_in = int(mask_slice.sum())

            slicename = f"/slice{islice + 1:06d}/"

            if n_in < min_particles:
                if template_slice is None:
                    raise RuntimeError(
                        "No valid template slice available for zero-current slice generation."
                    )
                sd = _make_zero_current_slice_from_template(template_slice)
            else:
                # ---- current ----
                if need_cdf and cdf_result is not None:
                    dF = float(Fx(slice_right)) - float(Fx(slice_left))
                    current = dF * Qb / (slicelength / g_c)
                else:
                    current = float(Q_abs[mask_slice].sum()) / (slicelength / g_c)

                # ---- build 6-D array for resampling ----
                z_sl = z[mask_slice]
                theta_raw = (z_sl - slice_left) / slicelength * _TWO_PI
                theta_raw = np.clip(theta_raw, 0.0, _TWO_PI * (1.0 - 1e-15))

                arr6d = np.column_stack([
                    x[mask_slice],
                    y[mask_slice],
                    theta_raw,
                    px_norm[mask_slice],
                    py_norm[mask_slice],
                    gamma[mask_slice],
                ])
                resampled = _resample_slice_particles(arr6d, npart, rng)

                x_out    = resampled[:, 0]
                y_out    = resampled[:, 1]
                theta_rs = resampled[:, 2]  # already folded to [0, 2π)
                px_out   = resampled[:, 3]
                py_out   = resampled[:, 4]
                gamma_out = resampled[:, 5]

                # ---- theta assignment ----
                if theta_method == "direct":
                    theta_out = theta_rs

                else:
                    # Generate uniform samples ∈ [0, 1)
                    if theta_method == "hammersley":
                        samp = halton_samples[halton_idx: halton_idx + npart]
                        halton_idx += npart
                    else:  # penman
                        Ne_sl = (current * slicelength / g_c / abs(g_e0)
                                 if current > 0 else float(npart))
                        r0 = (np.arange(1, npart + 1) - 0.5) / npart
                        delta_p = np.sqrt(3.0 * npart / max(Ne_sl, 1.0))
                        dr = (rng.random(npart) * 2.0 - 1.0) * delta_p / _TWO_PI
                        samp = (r0 + dr) % 1.0

                    # Map through CDF to theta
                    Fa = float(Fx(slice_left))
                    Fb = float(Fx(slice_right))
                    dF = Fb - Fa
                    if dF > 0.0:
                        z_new = Fx_inv(Fa + samp * dF).astype(float)
                        z_new = np.clip(z_new, slice_left, slice_right)
                        theta_new = (z_new - slice_left) / slicelength * _TWO_PI
                        theta_new = np.clip(theta_new, 0.0, _TWO_PI * (1.0 - 1e-15))
                    else:
                        theta_new = rng.random(npart) * _TWO_PI

                    # Sort-and-replace: preserve rank of resampled theta
                    rank = np.argsort(theta_rs)
                    theta_out = np.empty(npart)
                    theta_out[rank] = np.sort(theta_new)

                sd = dict(x=x_out, y=y_out, theta=theta_out,
                          px=px_out, py=py_out, gamma=gamma_out,
                          current=current)
                template_slice = dict(
                    x=np.asarray(x_out, dtype=float).copy(),
                    y=np.asarray(y_out, dtype=float).copy(),
                    px=np.asarray(px_out, dtype=float).copy(),
                    py=np.asarray(py_out, dtype=float).copy(),
                    gamma=np.asarray(gamma_out, dtype=float).copy(),
                )

            _write_dataset_with_unit(f, slicename + "current", [sd["current"]], "A")
            _write_dataset_with_unit(f, slicename + "x", sd["x"], "m")
            _write_dataset_with_unit(f, slicename + "y", sd["y"], "m")
            _write_dataset_with_unit(f, slicename + "theta", sd["theta"], "rad")
            _write_dataset_with_unit(f, slicename + "px", sd["px"], "rad")
            _write_dataset_with_unit(f, slicename + "py", sd["py"], "rad")
            _write_dataset_with_unit(f, slicename + "gamma", sd["gamma"], " ")

        # Root datasets
        _write_dataset_with_unit(f, "slicelength", [slicelength], "m")
        _write_dataset_with_unit(f, "slicespacing", [slicespacing], "m")
        _write_dataset_with_unit(f, "refposition", [s0], "m")
        _write_dataset_with_unit(f, "slicecount", [nslice])
        _write_dataset_with_unit(f, "beamletsize", [nbins])
        _write_dataset_with_unit(f, "one4one", [0])
        if version == 4:
            _write_dataset_with_unit(f, "/Meta/Version/Major", [4.0], " ")
            _write_dataset_with_unit(f, "/Meta/Version/Minor", [0.0], " ")
            _write_dataset_with_unit(f, "/Meta/Version/Revision", [0.0], " ")
            _write_dataset_with_unit(f, "/Meta/Version/Beta", [0.0], " ")
            f.create_dataset("/Meta/Version/Build_Info", data=np.bytes_("written by partdist"))
