from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from xtils import g_c, g_m0, g_e0

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
    m0: float,
    species_key: str,
    status_key: str,
) -> ParticleDistribution:
    vx, vy, vz = momentum_evc_to_velocity(
        np.array([ref.px]),
        np.array([ref.py]),
        np.array([ref.pz]),
        m0=m0,
    )

    return ParticleDistribution(
        x=np.array([ref.x], dtype=float),
        y=np.array([ref.y], dtype=float),
        z=np.array([ref.z], dtype=float),
        vx=vx,
        vy=vy,
        vz=vz,
        t=np.array([ref.t], dtype=float),
        Q=np.array([ref.Q], dtype=float),
        extras={
            species_key: ParticleArrayQuantity(
                name=species_key,
                data=np.array([ref.species], dtype=np.int64),
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
                data=np.array([ref.status], dtype=np.int64),
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
        vx=np.concatenate([first.vx, second.vx]),
        vy=np.concatenate([first.vy, second.vy]),
        vz=np.concatenate([first.vz, second.vz]),
        t=np.concatenate([first.t, second.t]),
        Q=np.concatenate([first.Q, second.Q]),
        extras=extras,
    )


def _build_reference_from_distribution(
    dist: ParticleDistribution,
    *,
    m0: float = g_m0,
    mode: str = "mean",
    weight: str | np.ndarray | None = "absQ",
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
    - "mean"  : use weighted means for x, y, z, vx, vy, vz
    - "zeros" : use zeros for x, y, z, vx, vy, vz
    """
    if mode == "mean":
        x0 = dist.mean("x", weight=weight)
        y0 = dist.mean("y", weight=weight)
        z0 = dist.mean("z", weight=weight)
        vx0 = dist.mean("vx", weight=weight)
        vy0 = dist.mean("vy", weight=weight)
        vz0 = dist.mean("vz", weight=weight)
        px0, py0, pz0 = velocity_to_momentum_evc(
            np.array([vx0]),
            np.array([vy0]),
            np.array([vz0]),
            m0=m0,
        )
        px0 = float(px0[0])
        py0 = float(py0[0])
        pz0 = float(pz0[0])
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
    m0: float = g_m0,
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
    - Momentum is converted to velocity.
    - Units in memory are:
        x, y, z   : m
        vx, vy, vz: m/s
        t         : s
        Q         : C

    Parameters
    ----------
    include_reference_particle
        If True, include the ASTRA reference particle itself as the first particle
        in the returned ParticleDistribution. Default is True.
    return_reference
        If True, also return the parsed AstraReferenceParticle.
    m0
        Rest mass used for momentum/velocity conversion.
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
            vx=np.empty(0, dtype=float),
            vy=np.empty(0, dtype=float),
            vz=np.empty(0, dtype=float),
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
        x = data[:, 0] + ref.x
        y = data[:, 1] + ref.y
        z = data[:, 2] + ref.z

        px = data[:, 3] + ref.px
        py = data[:, 4] + ref.py
        pz = data[:, 5] + ref.pz
        vx, vy, vz = momentum_evc_to_velocity(px, py, pz, m0=m0)

        t = data[:, 6] * 1.0e-9 + ref.t   # ns -> s, then add reference time
        Q = data[:, 7] * 1.0e-9           # nC -> C, absolute

        species = np.rint(data[:, 8]).astype(np.int64)
        status = np.rint(data[:, 9]).astype(np.int64)

        real_dist = ParticleDistribution(
            x=x,
            y=y,
            z=z,
            vx=vx,
            vy=vy,
            vz=vz,
            t=t,
            Q=Q,
            extras={
                species_key: ParticleArrayQuantity(
                    name=species_key,
                    data=species,
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
                    data=status,
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

    if include_reference_particle:
        ref_dist = _reference_to_distribution(
            ref,
            m0=m0,
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
    m0: float = g_m0,
    species_key: str = "species",
    status_key: str = "status",
    default_species: int = 1,
    default_status: int = 5,
    weight: str | np.ndarray | None = "absQ",
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
    m0
        Rest mass used for velocity/momentum conversion.
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

            px0, py0, pz0 = velocity_to_momentum_evc(
                np.array([dist.vx[0]]),
                np.array([dist.vy[0]]),
                np.array([dist.vz[0]]),
                m0=m0,
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
                x=float(dist.x[0]),
                y=float(dist.y[0]),
                z=float(dist.z[0]),
                px=float(px0[0]),
                py=float(py0[0]),
                pz=float(pz0[0]),
                t=float(dist.t[0]),
                Q=float(dist.Q[0]),
                species=species0,
                status=status0,
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
                m0=m0,
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

    px, py, pz = velocity_to_momentum_evc(
        dist_particles.vx,
        dist_particles.vy,
        dist_particles.vz,
        m0=m0,
    )

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