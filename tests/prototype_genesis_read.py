#!/usr/bin/env python3
"""
Prototype reading of GENESIS .h5 file.
"""
import h5py
import numpy as np
from pathlib import Path
from xtils import g_c, g_m0, g_e0

def read_genesis_prototype(filepath, drop_zero_charge=True):
    """
    Prototype implementation.
    Returns (x, y, z, px, py, pz, t, Q) arrays.
    """
    with h5py.File(filepath, 'r') as f:
        # Get global parameters
        slicelength = float(f['slicelength'][()].item())  # m
        slicespacing = float(f['slicespacing'][()].item())  # m
        refposition = float(f['refposition'][()].item())  # m
        beamletsize = int(f['beamletsize'][()].item())
        one4one = int(f['one4one'][()].item())
        slicecount = int(f['slicecount'][()].item())

        # Speed of light
        c = g_c  # m/s

        # List slice groups (exclude datasets like slicecount, slicelength, slicespacing)
        slice_keys = [k for k in f.keys() if k.startswith('slice') and isinstance(f[k], h5py.Group)]
        slice_keys.sort()  # ensure order
        # Should match slicecount (but sometimes differs)
        if len(slice_keys) != slicecount:
            print(f"Warning: slice count mismatch: keys {len(slice_keys)} vs slicecount {slicecount}")

        # Particles per slice (constant)
        # Get from first slice's x dataset
        first_slice = f[slice_keys[0]]
        n_per_slice = first_slice['x'].shape[0]

        # Pre-allocate arrays (will accumulate)
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
            # debug print removed
            try:
                ds_current = grp['current']
                current = float(ds_current[()][0])  # A
            except Exception as e:
                print(f"Error reading current for {sk}: {e}")
                # Try alternative access
                current = 0.0

            # Compute charge per particle in this slice
            # Q_slice = current * slicelength / c
            Q_slice = current * slicelength / c  # C
            Q_per_particle = Q_slice / n_per_slice  # C

            if drop_zero_charge and Q_per_particle == 0:
                # Skip this entire slice
                continue

            # Particle data
            x = grp['x'][:]  # m
            y = grp['y'][:]  # m
            gamma = grp['gamma'][:]  # dimensionless
            px_norm = grp['px'][:]  # dimensionless p_x/(m_e c)
            py_norm = grp['py'][:]  # dimensionless p_y/(m_e c)
            theta = grp['theta'][:]  # rad

            # Number of particles in this slice (should be n_per_slice)
            n_this = len(x)
            if n_this != n_per_slice:
                # This should not happen, but handle gracefully
                # Recompute Q_per_particle based on actual count
                Q_per_particle = Q_slice / n_this if n_this > 0 else 0.0

            # Compute pz from gamma and px, py
            # gamma^2 = 1 + px^2 + py^2 + pz^2
            # where px, py, pz are dimensionless p/(m_e c)
            pz_squared = gamma**2 - 1 - px_norm**2 - py_norm**2
            # Handle possible numerical errors (should be positive)
            pz_squared = np.maximum(pz_squared, 0.0)
            pz_norm = np.sqrt(pz_squared)
            # Assume pz positive (forward direction)

            # Convert dimensionless momentum to eV/c
            # Conversion factor: m_e * c^2 / e0 = 510998.946 eV
            factor_evc = g_m0 * g_c**2 / abs(g_e0)  # should be ~510998.946
            px_evc = px_norm * factor_evc
            py_evc = py_norm * factor_evc
            pz_evc = pz_norm * factor_evc

            # Compute z position of slice center
            # Assuming slices are equally spaced starting from refposition
            z_slice_center = refposition + i * slicespacing  # m

            # Assign same z to all particles in slice (coarse approximation)
            # Alternatively, could use theta to compute relative z,
            # but need undulator wavelength. For now use slice center.
            z = np.full(n_this, z_slice_center, dtype=float)

            # Time: assume t = z / c (approximation)
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
            return empty, empty, empty, empty, empty, empty, empty, empty

        # Concatenate
        x_arr = np.concatenate(all_x)
        y_arr = np.concatenate(all_y)
        z_arr = np.concatenate(all_z)
        px_arr = np.concatenate(all_px)
        py_arr = np.concatenate(all_py)
        pz_arr = np.concatenate(all_pz)
        t_arr = np.concatenate(all_t)
        Q_arr = np.concatenate(all_Q)

        return x_arr, y_arr, z_arr, px_arr, py_arr, pz_arr, t_arr, Q_arr

def test_prototype():
    filepath = Path(__file__).parent / 'data' / 'scan.000.out.par.h5'
    print("Reading with drop_zero_charge=True...")
    x, y, z, px, py, pz, t, Q = read_genesis_prototype(filepath, drop_zero_charge=True)
    print(f"Number of particles: {len(x)}")
    print(f"Charge per particle range: {Q.min():.3e} to {Q.max():.3e} C")
    print(f"Mean charge: {Q.mean():.3e} C")
    print(f"x range: {x.min():.3e} to {x.max():.3e} m")
    print(f"y range: {y.min():.3e} to {y.max():.3e} m")
    print(f"z range: {z.min():.3e} to {z.max():.3e} m")
    print(f"px range: {px.min():.3e} to {px.max():.3e} eV/c")
    print(f"py range: {py.min():.3e} to {py.max():.3e} eV/c")
    print(f"pz range: {pz.min():.3e} to {pz.max():.3e} eV/c")
    print(f"t range: {t.min():.3e} to {t.max():.3e} s")

    # Compute kinetic energy from gamma
    # gamma = sqrt(1 + p_norm^2)
    # p_norm = p_eVc / factor_evc
    factor_evc = g_m0 * g_c**2 / abs(g_e0)
    p_norm_sq = (px/factor_evc)**2 + (py/factor_evc)**2 + (pz/factor_evc)**2
    gamma = np.sqrt(1 + p_norm_sq)
    kinetic_energy_eV = (gamma - 1) * abs(g_e0)  # eV
    print(f"Kinetic energy range: {kinetic_energy_eV.min():.3e} to {kinetic_energy_eV.max():.3e} eV")
    print(f"Mean gamma: {gamma.mean():.3f}")

    print("\nReading with drop_zero_charge=False...")
    x2, y2, z2, px2, py2, pz2, t2, Q2 = read_genesis_prototype(filepath, drop_zero_charge=False)
    print(f"Number of particles: {len(x2)}")
    print(f"Zero-charge particles: {(Q2 == 0).sum()}")
    print(f"Non-zero charge particles: {(Q2 != 0).sum()}")

    # Verify that dropping zero charge removes exactly those particles
    mask_nonzero = Q2 != 0
    assert np.allclose(x2[mask_nonzero], x)
    assert np.allclose(y2[mask_nonzero], y)
    # z may differ due to slice ordering? Actually same slices kept.
    # Check that all zero-charge slices are removed
    # Count slices with zero current
    with h5py.File(filepath, 'r') as f:
        slice_keys = [k for k in f.keys() if k.startswith('slice')]
        slice_keys.sort()
        zero_current_slices = 0
        for sk in slice_keys:
            current = float(f[sk]['current'][()])
            if current == 0:
                zero_current_slices += 1
        particles_per_slice = f[slice_keys[0]]['x'].shape[0]
        expected_zero_particles = zero_current_slices * particles_per_slice
        actual_zero_particles = (Q2 == 0).sum()
        print(f"Expected zero-charge particles: {expected_zero_particles}")
        print(f"Actual zero-charge particles: {actual_zero_particles}")
        assert expected_zero_particles == actual_zero_particles

if __name__ == '__main__':
    test_prototype()