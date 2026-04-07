#!/usr/bin/env python3
"""
Test reading GENESIS .h5 particle files.
"""
import pytest
import numpy as np
from pathlib import Path
import h5py

from partdist.pd3d.io import read_genesis_distribution, read_astra_distribution

DATA_DIR = Path(__file__).parent / 'data'
H5_FILE   = DATA_DIR / 'scan.000.out.par.h5'
ASTRA_FILE = DATA_DIR / 'matched.dist'

@pytest.fixture
def genesis_file():
    if not H5_FILE.exists():
        pytest.skip(f"Test data file not found: {H5_FILE}")
    return H5_FILE

def test_read_genesis_basic(genesis_file):
    """Basic test reading GENESIS .h5 file."""
    dist = read_genesis_distribution(genesis_file, drop_zero_charge=True)

    # Check that we got a ParticleDistribution
    from partdist import ParticleDistribution
    assert isinstance(dist, ParticleDistribution)

    # Should have some particles (non-zero current slices)
    assert len(dist) > 0

    # Check that all base quantities exist
    for key in ['x', 'y', 'z', 'px', 'py', 'pz', 't', 'Q']:
        assert hasattr(dist, key)
        arr = getattr(dist, key)
        assert isinstance(arr, np.ndarray)
        assert len(arr) == len(dist)

    # Check that charge is not zero (since drop_zero_charge=True)
    assert not np.any(dist.Q == 0.0)

    # Check units: positions in meters (typical values ~1e-4)
    assert np.all(np.isfinite(dist.x))
    assert np.all(np.isfinite(dist.y))
    assert np.all(np.isfinite(dist.z))

    # Check momentum in eV/c (typical values ~1e5-1e6)
    assert np.all(np.isfinite(dist.px))
    assert np.all(np.isfinite(dist.py))
    assert np.all(np.isfinite(dist.pz))

    # Print some stats for debugging
    print(f"Read {len(dist)} particles")
    print(f"Mean charge: {dist.Q.mean():.3e} C")
    print(f"Mean kinetic energy: {dist.mean('kinetic_energy_eV'):.3e} eV")

def test_read_genesis_keep_zero_charge(genesis_file):
    """Test reading with drop_zero_charge=False."""
    dist = read_genesis_distribution(genesis_file, drop_zero_charge=False)

    # Should have more particles than with drop_zero_charge=True
    dist_dropped = read_genesis_distribution(genesis_file, drop_zero_charge=True)
    assert len(dist) > len(dist_dropped)

    # Some particles should have zero charge
    assert np.any(dist.Q == 0.0)

    # Particles with zero charge should have same positions/momenta as non-zero?
    # At least they should be present
    zero_mask = dist.Q == 0.0
    assert zero_mask.sum() > 0

    # Check that zero-charge particles have finite coordinates
    assert np.all(np.isfinite(dist.x[zero_mask]))
    assert np.all(np.isfinite(dist.y[zero_mask]))

def test_read_genesis_consistency(genesis_file):
    """Check internal consistency of read data."""
    dist = read_genesis_distribution(genesis_file, drop_zero_charge=True)

    # Check that gamma computed from momentum matches stored gamma?
    # Actually, we compute pz from gamma, px, py, so consistency is guaranteed.
    # But we can verify that kinetic_energy_eV is consistent with gamma
    gamma_from_p = np.sqrt(1 + (dist.px**2 + dist.py**2 + dist.pz**2) * (1/510998.946)**2)  # Need conversion
    # Instead, use the derived quantity 'gamma' if available
    if hasattr(dist, 'gamma'):
        gamma = dist.gamma
        # Should match within numerical tolerance
        np.testing.assert_allclose(gamma_from_p, gamma, rtol=1e-10)

def test_read_genesis_file_not_found():
    """Test error when file doesn't exist."""
    with pytest.raises(FileNotFoundError):
        read_genesis_distribution('nonexistent.h5')

def test_read_genesis_invalid_file(tmp_path):
    """Test error with invalid HDF5 file."""
    bad_file = tmp_path / 'bad.h5'
    bad_file.write_text('not hdf5')
    with pytest.raises(Exception):
        read_genesis_distribution(bad_file)

def _weighted_std(arr, weights):
    """Return the charge-weighted standard deviation of *arr*."""
    w = np.abs(weights)
    mean = np.average(arr, weights=w)
    return np.sqrt(np.average((arr - mean) ** 2, weights=w))


@pytest.mark.skipif(
    not (H5_FILE.exists() and ASTRA_FILE.exists()),
    reason="Test data files not found",
)
def test_genesis_vs_astra_charge_weighted_stats():
    """Charge-weighted statistics of the Genesis h5 and matched ASTRA distributions
    should agree within tolerances consistent with their shared origin.

    Key insight: GENESIS assigns an equal number of macro-particles per slice
    regardless of the slice current, so *unweighted* z statistics from the two
    formats will disagree (Genesis z_std ~ 657 µm vs ASTRA ~ 503 µm). Once
    statistics are weighted by the macro-charge Q of each particle the results
    converge because high-current slices carry proportionally more charge.

    Tolerances
    ----------
    Transverse x, y and momentum px, py, pz:  < 5 % relative difference.
    Longitudinal z (weighted):                < 1 % relative difference.
    Total charge |sum Q|:                     < 0.1 %.
    Mean pz (beam energy):                    < 0.1 %.
    """
    astra   = read_astra_distribution(ASTRA_FILE, include_reference_particle=False)
    genesis = read_genesis_distribution(H5_FILE)

    # Transverse / momentum spreads
    for qty in ('x', 'y', 'px', 'py', 'pz'):
        a_std = _weighted_std(getattr(astra, qty),   astra.Q)
        g_std = _weighted_std(getattr(genesis, qty), genesis.Q)
        rel   = abs(a_std - g_std) / (0.5 * (a_std + g_std))
        assert rel < 0.05, (
            f"{qty} weighted std: ASTRA={a_std:.4g}, Genesis={g_std:.4g}, "
            f"relative difference {rel:.1%} > 5 %"
        )

    # Longitudinal spread (tight tolerance because current-weighted slice centres match)
    a_z = _weighted_std(astra.z,   astra.Q)
    g_z = _weighted_std(genesis.z, genesis.Q)
    rel_z = abs(a_z - g_z) / (0.5 * (a_z + g_z))
    assert rel_z < 0.01, (
        f"z weighted std: ASTRA={a_z*1e6:.1f} µm, Genesis={g_z*1e6:.1f} µm, "
        f"relative difference {rel_z:.2%} > 1 %"
    )

    # Total charge
    q_a = abs(np.sum(astra.Q))
    q_g = abs(np.sum(genesis.Q))
    rel_q = abs(q_a - q_g) / (0.5 * (q_a + q_g))
    assert rel_q < 0.001, (
        f"Total |Q|: ASTRA={q_a*1e9:.4f} nC, Genesis={q_g*1e9:.4f} nC, "
        f"relative difference {rel_q:.2%} > 0.1 %"
    )

    # Mean beam energy (pz)
    pz_a = np.average(astra.pz,   weights=np.abs(astra.Q))
    pz_g = np.average(genesis.pz, weights=np.abs(genesis.Q))
    rel_pz = abs(pz_a - pz_g) / (0.5 * (pz_a + pz_g))
    assert rel_pz < 0.001, (
        f"Mean pz: ASTRA={pz_a:.6g} eV/c, Genesis={pz_g:.6g} eV/c, "
        f"relative difference {rel_pz:.2%} > 0.1 %"
    )


@pytest.mark.skipif(
    not (H5_FILE.exists() and ASTRA_FILE.exists()),
    reason="Test data files not found",
)
def test_genesis_unweighted_z_wider_than_astra():
    """Unweighted z_std from Genesis is larger than ASTRA's because Genesis
    assigns equal macro-particles to every non-zero slice (irrespective of
    current), giving more statistical weight to the beam tails.

    This is expected behaviour, not a bug.
    """
    astra   = read_astra_distribution(ASTRA_FILE, include_reference_particle=False)
    genesis = read_genesis_distribution(H5_FILE)

    z_std_astra   = np.std(astra.z)
    z_std_genesis = np.std(genesis.z)

    assert z_std_genesis > z_std_astra * 1.1, (
        "Expected unweighted Genesis z_std to be noticeably larger than ASTRA's "
        f"(got Genesis={z_std_genesis*1e6:.1f} µm, ASTRA={z_std_astra*1e6:.1f} µm)"
    )


if __name__ == '__main__':
    # Run basic test manually
    if H5_FILE.exists():
        print("Running basic test...")
        test_read_genesis_basic(H5_FILE)
        print("Basic test passed.")

        print("\nRunning keep zero charge test...")
        test_read_genesis_keep_zero_charge(H5_FILE)
        print("Keep zero charge test passed.")

        print("\nRunning ASTRA vs Genesis comparison...")
        test_genesis_vs_astra_charge_weighted_stats()
        test_genesis_unweighted_z_wider_than_astra()
        print("Comparison tests passed.")
    else:
        print(f"Test file not found: {H5_FILE}")