#!/usr/bin/env python3
"""
Test that z coordinate calculation from theta is correct.
"""
import numpy as np
import h5py
from pathlib import Path
from partdist.pd3d.io import read_genesis_distribution

def test_z_calculation():
    filepath = Path(__file__).parent / 'data' / 'scan.000.out.par.h5'
    if not filepath.exists():
        print(f"Test file not found: {filepath}")
        return

    # Read using our function
    dist = read_genesis_distribution(filepath, drop_zero_charge=False)

    # Also read raw data to compute expected z
    with h5py.File(filepath, 'r') as f:
        slicelength = float(f['slicelength'][()].item())
        slicespacing = float(f['slicespacing'][()].item())
        refposition = float(f['refposition'][()].item())

        # Get slice groups
        slice_keys = [k for k in f.keys() if k.startswith('slice') and isinstance(f[k], h5py.Group)]
        slice_keys.sort()

        # For each slice, compute expected z from theta
        expected_z_parts = []
        particle_count = 0

        for i, sk in enumerate(slice_keys):
            grp = f[sk]
            current = float(grp['current'][()][0])
            if current == 0:
                # Skip zero-current slices? Keep for comprehensive test
                pass
            theta = grp['theta'][:]  # rad
            n_this = len(theta)

            # Compute expected z for this slice
            z_relative = (theta / (2 * np.pi) + 0.5) * slicelength
            z_expected = z_relative + refposition + i * slicespacing
            expected_z_parts.append(z_expected)
            particle_count += n_this

        # Concatenate all expected z
        expected_z_all = np.concatenate(expected_z_parts)

    # Compare with z from ParticleDistribution
    z_from_dist = dist.z

    # They should match exactly (same order of slices)
    np.testing.assert_allclose(z_from_dist, expected_z_all, rtol=1e-12)
    print(f"✓ z coordinate calculation matches for {len(z_from_dist)} particles")

    # Check range for a few slices
    with h5py.File(filepath, 'r') as f:
        # Check first non-zero current slice
        for i, sk in enumerate(slice_keys[:5]):
            grp = f[sk]
            current = float(grp['current'][()][0])
            if current == 0:
                continue
            theta = grp['theta'][:]
            z_relative = (theta / (2 * np.pi) + 0.5) * slicelength
            z_min_expected = refposition + i * slicespacing + z_relative.min()
            z_max_expected = refposition + i * slicespacing + z_relative.max()

            # Since z_relative ∈ [0, slicelength]
            assert z_relative.min() >= 0, f"z_relative.min() = {z_relative.min()} < 0"
            assert z_relative.max() <= slicelength, f"z_relative.max() = {z_relative.max()} > slicelength"

            print(f"Slice {sk}: theta range [{theta.min():.3f}, {theta.max():.3f}] rad")
            print(f"  z_relative range [{z_relative.min():.2e}, {z_relative.max():.2e}] m")
            print(f"  z range [{z_min_expected:.2e}, {z_max_expected:.2e}] m")

    # Print overall statistics
    print(f"\nOverall z range: [{z_from_dist.min():.2e}, {z_from_dist.max():.2e}] m")
    print(f"Mean z: {z_from_dist.mean():.2e} m")
    print(f"Standard deviation of z: {z_from_dist.std():.2e} m")

    # Verify that particles are sorted by slice (z should increase overall)
    # Actually, within each slice z varies, but overall trend should increase
    # Simple check: mean z per slice should increase with slice index
    with h5py.File(filepath, 'r') as f:
        slice_means = []
        for i, sk in enumerate(slice_keys):
            grp = f[sk]
            current = float(grp['current'][()][0])
            if current == 0:
                continue
            theta = grp['theta'][:]
            z_relative = (theta / (2 * np.pi) + 0.5) * slicelength
            z_slice = z_relative + refposition + i * slicespacing
            slice_means.append(z_slice.mean())

        # Check monotonic increase (allow small tolerance for floating point)
        for j in range(1, len(slice_means)):
            assert slice_means[j] > slice_means[j-1] - 1e-12, \
                f"Slice {j} mean z {slice_means[j]:.2e} <= previous {slice_means[j-1]:.2e}"

    print("✓ All tests passed")

if __name__ == '__main__':
    test_z_calculation()