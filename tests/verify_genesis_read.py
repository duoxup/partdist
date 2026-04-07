#!/usr/bin/env python3
"""
Verify the new read_genesis_distribution function.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from partdist.pd3d.io import read_genesis_distribution

def main():
    filepath = Path(__file__).parent / 'data' / 'scan.000.out.par.h5'
    if not filepath.exists():
        print(f"Test file not found: {filepath}")
        return

    print("Testing read_genesis_distribution with drop_zero_charge=True...")
    dist = read_genesis_distribution(filepath, drop_zero_charge=True)
    print(f"  Number of particles: {len(dist)}")
    print(f"  Total charge: {dist.Q.sum():.3e} C")
    print(f"  Mean x: {dist.mean('x'):.3e} m")
    print(f"  Mean y: {dist.mean('y'):.3e} m")
    print(f"  Mean z: {dist.mean('z'):.3e} m")
    print(f"  Mean kinetic energy: {dist.mean('kinetic_energy_eV'):.3e} eV")
    print(f"  Mean gamma: {dist.mean('gamma'):.3f}")
    print(f"  Charge per particle range: {dist.Q.min():.3e} to {dist.Q.max():.3e} C")
    assert len(dist) > 0
    assert dist.Q.min() > 0  # all charges positive

    print("\nTesting read_genesis_distribution with drop_zero_charge=False...")
    dist2 = read_genesis_distribution(filepath, drop_zero_charge=False)
    print(f"  Number of particles: {len(dist2)}")
    print(f"  Zero-charge particles: {(dist2.Q == 0).sum()}")
    print(f"  Non-zero charge particles: {(dist2.Q != 0).sum()}")
    assert len(dist2) > len(dist)
    assert (dist2.Q == 0).sum() > 0

    # Verify that dropping zero charge removes exactly those particles
    mask_nonzero = dist2.Q != 0
    # The order of particles should be the same (slices concatenated in order)
    # We can check that the first N particles match
    n_nonzero = mask_nonzero.sum()
    # Since slices with zero current are skipped, the remaining slices appear in same order
    # Therefore, the first n_nonzero particles in dist2 (with nonzero charge) should match dist
    # But we need to compare carefully. For simplicity, check that total charge matches.
    total_charge_nonzero = dist2.Q[mask_nonzero].sum()
    total_charge_dropped = dist.Q.sum()
    assert abs(total_charge_nonzero - total_charge_dropped) < 1e-20

    print("\nAll checks passed.")
    return dist

if __name__ == '__main__':
    from partdist.pd3d.viz import hist2d_pd3d
    dist  = main()
    hist2d_pd3d(dist, x='z', y='pz',
                color_threshold=1e-2,
                cmap='jet')