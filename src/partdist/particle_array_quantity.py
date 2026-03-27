from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


QuantityDTypeKind = Literal["float", "int", "bool", "category"]
QuantityCategory = Literal[
    "position",
    "velocity",
    "momentum",
    "time",
    "charge",
    "energy",
    "current",
    "geometry",
    "flag",
    "other",
]


@dataclass
class ParticleArrayQuantity:
    """
    One particle-resolved quantity bound to all macro-particles.

    Parameters
    ----------
    name
        Internal programmatic name, e.g. 'x', 'pz', 'status'.
    data
        1D array of length N.
    unit
        Physical unit string, e.g. 'm', 'm/s', 'eV/c', 'C'.
        Use '' for dimensionless or categorical quantities.
    dtype_kind
        Semantic data type category.
    short_name
        Short display name.
    long_name
        Human-readable long name.
    latex_name
        LaTeX-style label string.
    category
        High-level physical category.
    is_derived
        Whether this quantity is derived from other stored quantities.
    is_discrete
        Whether the quantity should be treated as discrete in plotting.
    preferred_scale
        Plotting hint, e.g. 'linear' or 'log'.
    """

    name: str
    data: np.ndarray
    unit: str = ""
    dtype_kind: QuantityDTypeKind = "float"
    short_name: str | None = None
    long_name: str | None = None
    latex_name: str | None = None
    category: QuantityCategory = "other"
    is_derived: bool = False
    is_discrete: bool | None = None
    preferred_scale: str = "linear"

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data).reshape(-1)

        if self.data.ndim != 1:
            raise ValueError(f"{self.name!r} must be a 1D array, got shape {self.data.shape}.")

        if self.dtype_kind == "float":
            self.data = self.data.astype(float, copy=False)
        elif self.dtype_kind == "int":
            self.data = self.data.astype(np.int64, copy=False)
        elif self.dtype_kind == "bool":
            self.data = self.data.astype(bool, copy=False)

        if self.short_name is None:
            self.short_name = self.name
        if self.long_name is None:
            self.long_name = self.name
        if self.latex_name is None:
            self.latex_name = self.short_name

        if self.is_discrete is None:
            self.is_discrete = self.dtype_kind in ("int", "bool", "category")

    @property
    def n(self) -> int:
        return self.data.size

    def copy(self) -> "ParticleArrayQuantity":
        return ParticleArrayQuantity(
            name=self.name,
            data=self.data.copy(),
            unit=self.unit,
            dtype_kind=self.dtype_kind,
            short_name=self.short_name,
            long_name=self.long_name,
            latex_name=self.latex_name,
            category=self.category,
            is_derived=self.is_derived,
            is_discrete=self.is_discrete,
            preferred_scale=self.preferred_scale,
        )