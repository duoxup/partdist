#!/usr/bin/env python3
"""
Check GENESIS .h5 file for units and conversions.
"""
import h5py
import numpy as np
from pathlib import Path

def check_units(filepath):
    with h5py.File(filepath, 'r') as f:
        print("Global datasets:")
        for ds_name in ['beamletsize', 'one4one', 'refposition', 'slicecount', 'slicelength', 'slicespacing']:
            if ds_name in f:
                val = f[ds_name][()]
                print(f"  {ds_name}: {val} (shape {val.shape}, dtype {val.dtype})")

        # Check Meta/Version
        if 'Meta' in f and 'Version' in f['Meta']:
            ver = f['Meta/Version']
            for k in ver.keys():
                print(f"  Meta/Version/{k}: {ver[k][:]}")

        # Examine a few slices to see relationship between current and particle data
        slice_keys = [k for k in f.keys() if k.startswith('slice')]
        # Pick first, middle, and last slice
        indices = [0, len(slice_keys)//2, -1]
        for idx in indices:
            sk = slice_keys[idx]
            grp = f[sk]
            current = grp['current'][0]
            print(f"\n{sk}: current = {current:.6e} A")
            # Compute mean of particle data
            for key in ['x', 'y', 'gamma', 'px', 'py', 'theta']:
                data = grp[key][:]
                print(f"  {key}: mean={data.mean():.6e}, std={data.std():.6e}")

        # Check if theta is in radians (should be between -pi and pi)
        # Take first slice
        sk = slice_keys[0]
        theta = f[sk]['theta'][:]
        print(f"\nTheta range in {sk}: min={theta.min():.6e}, max={theta.max():.6e}")
        print(f"  Expected range for radians: -π to π (-3.1416 to 3.1416)")

        # Check if px, py are dimensionless (typical values ~1e-3?)
        px = f[sk]['px'][:]
        py = f[sk]['py'][:]
        print(f"px range: min={px.min():.6e}, max={px.max():.6e}, mean={px.mean():.6e}")
        print(f"py range: min={py.min():.6e}, max={py.max():.6e}, mean={py.mean():.6e}")

        # Check gamma values (should be ~78.6 from earlier)
        gamma = f[sk]['gamma'][:]
        print(f"gamma range: min={gamma.min():.6e}, max={gamma.max():.6e}")

        # Calculate momentum p = gamma * beta = gamma * v/c
        # But we need beta. Actually, px, py might be gamma * beta_x, gamma * beta_y?
        # Let's compute beta from gamma assuming p = gamma * beta
        # However, we need pz. Not directly given. Maybe theta is related to z?
        # In FEL, theta = 2*pi*(z - v_z*t)/lambda_w, where lambda_w is undulator period.
        # Not straightforward.

        # Let's compute total charge in a slice: Q_slice = current * (slicelength / c) ?
        # Actually, current I = dQ/dt, but in beam frame, I = Q * f * bunching?
        # For relativistic beam, current I = Q * c / L_slice ? Not sure.

        # Look for documentation in attributes
        print("\nChecking attributes...")
        for key in f.attrs:
            print(f"  root attr {key}: {f.attrs[key]}")

        if 'Meta' in f:
            for key in f['Meta'].attrs:
                print(f"  Meta attr {key}: {f['Meta'].attrs[key]}")

if __name__ == '__main__':
    filepath = Path(__file__).parent / 'data' / 'scan.000.out.par.h5'
    check_units(filepath)