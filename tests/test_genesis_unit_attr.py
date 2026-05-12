"""Regression test: Genesis writer must use empty unit for dimensionless px/py columns."""
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

from partdist import ParticleDistribution3D
from partdist.pd3d.io import write_genesis_distribution


def _make_tiny_dist(n=64):
    rng = np.random.default_rng(0)
    # pz centered at ~100 MeV/c (relativistic, ~200 in gamma units)
    pz_center = 1e8  # eV/c
    return ParticleDistribution3D(
        x=rng.normal(0, 1e-4, n),
        y=rng.normal(0, 1e-4, n),
        z=rng.normal(0, 1e-6, n),
        px=rng.normal(0, 1e5, n),
        py=rng.normal(0, 1e5, n),
        pz=pz_center + rng.normal(0, 1e5, n),
        t=np.zeros(n),
        Q=np.full(n, -1.6e-19),
    )


def test_genesis_writer_px_py_have_empty_unit():
    """px/py datasets store dimensionless p/(m_e c); unit attribute must be empty."""
    d = _make_tiny_dist()
    # lambda0: radiation wavelength [m], s0: start of window [m], slen: window length [m]
    # Use a 1-nm wavelength and a window that covers the distribution
    lambda0 = 1e-9
    s0 = -5e-6
    slen = 10e-6
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "tiny.h5"
        write_genesis_distribution(
            d,
            str(out),
            lambda0=lambda0,
            s0=s0,
            slen=slen,
            npart=8,
            on_warning="silent",
        )
        with h5py.File(out, "r") as f:
            slice_keys = [k for k in f.keys() if k.startswith("slice")]
            assert slice_keys, "expected at least one slice group"
            for sk in slice_keys[:1]:  # one slice is enough
                px_unit = f[sk + "/px"].attrs.get("unit", None)
                py_unit = f[sk + "/py"].attrs.get("unit", None)
                # Decode if h5py returned bytes
                if isinstance(px_unit, bytes):
                    px_unit = px_unit.decode()
                if isinstance(py_unit, bytes):
                    py_unit = py_unit.decode()
                assert px_unit == "", f"expected px unit='' (dimensionless), got {px_unit!r}"
                assert py_unit == "", f"expected py unit='' (dimensionless), got {py_unit!r}"
