#!/usr/bin/env python3
"""
Explore GENESIS .h5 particle file structure.
"""
import h5py
import numpy as np
from pathlib import Path

def explore_h5(filepath):
    with h5py.File(filepath, 'r') as f:
        print("Root groups/datasets:")
        for key in f.keys():
            print(f"  {key}: {type(f[key])}")

        # Check metadata
        if 'Meta' in f:
            print("\nMeta group:")
            for k, v in f['Meta'].items():
                print(f"  {k}: {v}")

        # Check global datasets
        for ds in ['beamletsize', 'one4one', 'refposition']:
            if ds in f:
                print(f"{ds}: {f[ds][:]}")

        # List slice groups
        slice_keys = [k for k in f.keys() if k.startswith('slice')]
        print(f"\nNumber of slice groups: {len(slice_keys)}")

        # Examine first slice
        if slice_keys:
            first = slice_keys[0]
            print(f"\nFirst slice: {first}")
            grp = f[first]
            for key in grp.keys():
                ds = grp[key]
                print(f"  {key}: shape {ds.shape}, dtype {ds.dtype}")
                # Print a few values
                if ds.ndim == 1 and ds.shape[0] > 0:
                    print(f"    min={ds[:].min():.6e}, max={ds[:].max():.6e}, mean={ds[:].mean():.6e}")
                elif ds.ndim == 0 or (ds.ndim == 1 and ds.shape[0] == 1):
                    print(f"    value={ds[()]}")

        # Examine a slice with possibly zero current
        # Look for slices where current is zero
        zero_current_slices = []
        for sk in slice_keys[:100]:  # limit to first 100 slices
            current = f[sk]['current'][()]
            if current == 0:
                zero_current_slices.append(sk)
        if zero_current_slices:
            print(f"\nSlices with zero current (first 10): {zero_current_slices[:10]}")
            # Check particle data in a zero-current slice
            sk = zero_current_slices[0]
            grp = f[sk]
            print(f"\nData in zero-current slice {sk}:")
            for key in ['x', 'y', 'gamma', 'px', 'py', 'theta']:
                ds = grp[key]
                data = ds[:]
                print(f"  {key}: min={data.min():.6e}, max={data.max():.6e}, mean={data.mean():.6e}")
        else:
            print("\nNo zero-current slices found in first 100 slices")

        # Check particle count consistency
        particle_counts = []
        for sk in slice_keys[:10]:
            grp = f[sk]
            if 'x' in grp:
                particle_counts.append(grp['x'].shape[0])
        if particle_counts:
            print(f"\nParticle counts per slice (first 10): {particle_counts}")
            if len(set(particle_counts)) == 1:
                print(f"All slices have same particle count: {particle_counts[0]}")
            else:
                print("WARNING: Particle counts vary across slices")

if __name__ == '__main__':
    filepath = Path(__file__).parent / 'data' / 'scan.000.out.par.h5'
    explore_h5(filepath)