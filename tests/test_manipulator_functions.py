#!/usr/bin/env python3
"""
Test script for manipulator.py functions.
Tests replicate_longitudinally, multiply_longitudinal_profile, set_linear_chirp, and match_twiss_xy.
Uses input distribution from /mnt/e/pitz/debug/gen.dist.
Saves output distributions to tests/data/ directory.
"""

import os
import numpy as np
from pathlib import Path

# Add parent directory to path if needed
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from partdist.pd3d.io import read_astra_distribution, write_astra_distribution
from partdist.pd3d.manipulator import (
    replicate_longitudinally,
    multiply_longitudinal_profile,
    set_linear_chirp,
    match_twiss_xy,
)

def main():
    print("Testing manipulator.py functions")

    # Input file path
    input_path = Path("/mnt/e/pitz/debug/gen.dist")
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        return 1

    # Output directory
    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Read input distribution
    print(f"Reading input distribution: {input_path}")
    try:
        dist = read_astra_distribution(input_path)
    except Exception as e:
        print(f"ERROR: Failed to read distribution: {e}")
        return 1

    print(f"Number of particles: {len(dist)}")
    print(f"Quantity keys: {dist.quantity_keys}")
    print(f"Centroid z: {dist.mean('z'):.6e} m")
    print(f"Centroid pz: {dist.mean('pz'):.6e} eV/c")
    print(f"Mean kinetic energy: {np.mean(dist.kinetic_energy_eV):.6e} eV")
    print()

    # Save original distribution for reference
    orig_output = output_dir / "original.dist"
    write_astra_distribution(orig_output, dist)
    print(f"Saved original distribution to: {orig_output}")

    # Test 1: replicate_longitudinally
    print("\n" + "="*60)
    print("Test 1: replicate_longitudinally")
    print("="*60)
    try:
        n_bunches = 3
        spacing = 0.001  # 1 mm spacing in z
        print(f"Parameters: n_bunches={n_bunches}, spacing={spacing} m, spacing_mode='z'")
        dist_replicated = replicate_longitudinally(
            dist,
            n_bunches=n_bunches,
            spacing=spacing,
            spacing_mode="z",
            charge_mode="preserve_per_bunch",
            add_copy_index=True,
            sort_by="z"
        )
        print(f"Replicated distribution particles: {len(dist_replicated)}")
        print(f"Copy index range: [{np.min(dist_replicated.get_data('copy_index'))}, {np.max(dist_replicated.get_data('copy_index'))}]")

        # Save result
        rep_output = output_dir / "replicated.dist"
        write_astra_distribution(rep_output, dist_replicated)
        print(f"Saved replicated distribution to: {rep_output}")
    except Exception as e:
        print(f"ERROR in replicate_longitudinally: {e}")
        import traceback
        traceback.print_exc()

    # Test 2: multiply_longitudinal_profile
    print("\n" + "="*60)
    print("Test 2: multiply_longitudinal_profile")
    print("="*60)
    try:
        # Define a Gaussian curve as sampled points
        z_centroid = dist.mean("z")
        z_std = dist.std("z")
        print(f"z centroid: {z_centroid:.6e} m, z std: {z_std:.6e} m")

        # Create Gaussian curve centered at beam centroid
        z_samples = np.linspace(z_centroid - 3*z_std, z_centroid + 3*z_std, 101)
        # Gaussian with sigma = z_std, amplitude 1.0
        gaussian = np.exp(-0.5 * ((z_samples - z_centroid) / z_std) ** 2)

        curve = (z_samples, gaussian)
        print(f"Gaussian curve: {len(z_samples)} samples, sigma={z_std:.6e} m")

        dist_multiplied = multiply_longitudinal_profile(
            dist,
            curve=curve,
            coordinate="z",
            center="mean",  # shift to beam centroid
            normalize=True,  # preserve total charge
            allow_negative=False,
            outside_value=0.0,
            inplace=False
        )

        # Check charge conservation
        Q_orig = np.sum(dist.Q)
        Q_new = np.sum(dist_multiplied.Q)
        print(f"Original total charge: {Q_orig:.6e} C")
        print(f"New total charge: {Q_new:.6e} C")
        print(f"Charge ratio (should be ~1.0): {Q_new/Q_orig:.6f}")

        # Save result
        mult_output = output_dir / "multiplied.dist"
        write_astra_distribution(mult_output, dist_multiplied)
        print(f"Saved multiplied distribution to: {mult_output}")
    except Exception as e:
        print(f"ERROR in multiply_longitudinal_profile: {e}")
        import traceback
        traceback.print_exc()

    # Test 3: set_linear_chirp
    print("\n" + "="*60)
    print("Test 3: set_linear_chirp")
    print("="*60)
    try:
        slope = 1e6  # 1e6 eV/c per meter
        print(f"Parameters: slope={slope:.3e} eV/c/m, center_x=True, preserve_mean_kinetic_energy=True")

        dist_chirped = set_linear_chirp(
            dist,
            slope=slope,
            intercept=None,  # let function determine intercept
            center_x=True,   # center z before applying slope
            center_y=True,   # anchor baseline near mean pz
            residual_mode="preserve",
            residual_scale=1.0,
            original_trend=None,
            fit_original_if_missing=True,
            preserve_mean_kinetic_energy=True,  # Try to preserve mean kinetic energy
            weight_for_centroid="Q",
            weight_for_energy="Q",
            inplace=False
        )

        # Check mean kinetic energy preservation
        ke_orig = np.mean(dist.kinetic_energy_eV)
        ke_new = np.mean(dist_chirped.kinetic_energy_eV)
        print(f"Original mean KE: {ke_orig:.6e} eV")
        print(f"New mean KE: {ke_new:.6e} eV")
        print(f"KE ratio: {ke_new/ke_orig:.6f}")

        # Check chirp slope
        # Simple linear fit to pz vs z
        z = dist_chirped.z
        pz = dist_chirped.pz
        # Weighted linear fit using |Q| as weights
        weights = np.abs(dist_chirped.Q)
        if np.sum(weights) > 0:
            z_mean = np.average(z, weights=weights)
            pz_mean = np.average(pz, weights=weights)
            cov = np.average((z - z_mean) * (pz - pz_mean), weights=weights)
            var_z = np.average((z - z_mean)**2, weights=weights)
            fitted_slope = cov / var_z if var_z > 0 else 0.0
            print(f"Fitted slope from distribution: {fitted_slope:.3e} eV/c/m")
            print(f"Target slope: {slope:.3e} eV/c/m")

        # Save result
        chirp_output = output_dir / "chirped.dist"
        write_astra_distribution(chirp_output, dist_chirped)
        print(f"Saved chirped distribution to: {chirp_output}")
    except Exception as e:
        print(f"ERROR in set_linear_chirp: {e}")
        import traceback
        traceback.print_exc()

    # Test 4: match_twiss_xy
    print("\n" + "="*60)
    print("Test 4: match_twiss_xy")
    print("="*60)
    try:
        # Target Twiss parameters
        alpha_x = 0.5
        beta_x = 0.1  # meters
        alpha_y = -0.3
        beta_y = 0.2  # meters

        print(f"Target Twiss parameters:")
        print(f"  x-plane: alpha={alpha_x}, beta={beta_x} m")
        print(f"  y-plane: alpha={alpha_y}, beta={beta_y} m")

        dist_matched = match_twiss_xy(
            dist,
            alpha_x=alpha_x,
            beta_x=beta_x,
            alpha_y=alpha_y,
            beta_y=beta_y,
            weight='Q',
            mask=None,
            center_before_match=True,
            preserve_centroid=True,
            inplace=False
        )

        # Compute resulting Twiss parameters to verify match
        from partdist.pd3d.analysis import compute_twiss_plane
        try:
            twiss_x = compute_twiss_plane(dist_matched, plane='x', weight='Q')
            twiss_y = compute_twiss_plane(dist_matched, plane='y', weight='Q')
            print(f"Resulting Twiss parameters:")
            print(f"  x-plane: alpha={twiss_x.alpha:.6f}, beta={twiss_x.beta:.6f} m, eps_geom={twiss_x.geometric_emittance:.6e} m·rad, eps_norm={twiss_x.normalized_emittance:.6e} m·rad")
            print(f"  y-plane: alpha={twiss_y.alpha:.6f}, beta={twiss_y.beta:.6f} m, eps_geom={twiss_y.geometric_emittance:.6e} m·rad, eps_norm={twiss_y.normalized_emittance:.6e} m·rad")
            print(f"  Target x beta: {beta_x} m, achieved: {twiss_x.beta:.6f} m")
            print(f"  Target y beta: {beta_y} m, achieved: {twiss_y.beta:.6f} m")
        except Exception as e_analysis:
            print(f"Note: Could not compute resulting Twiss parameters: {e_analysis}")

        # Save result
        match_output = output_dir / "matched.dist"
        write_astra_distribution(match_output, dist_matched)
        print(f"Saved matched distribution to: {match_output}")
    except Exception as e:
        print(f"ERROR in match_twiss_xy: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*60)
    print("All tests completed.")
    print(f"Output files saved in: {output_dir}")
    print("="*60)
    return 0

if __name__ == "__main__":
    sys.exit(main())