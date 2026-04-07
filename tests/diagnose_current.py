#!/usr/bin/env python3
import h5py
from pathlib import Path

filepath = Path(__file__).parent / 'data' / 'scan.000.out.par.h5'
with h5py.File(filepath, 'r') as f:
    sk = 'slice000001'
    grp = f[sk]
    ds = grp['current']
    print(f"Dataset: {ds}")
    print(f"Shape: {ds.shape}")
    print(f"Dtype: {ds.dtype}")
    print(f"Type: {type(ds)}")
    print(f"attrs: {list(ds.attrs)}")
    # Try different access methods
    try:
        val1 = ds[()]
        print(f"ds[()] = {val1}, type={type(val1)}")
    except Exception as e:
        print(f"ds[()] error: {e}")
    try:
        val2 = ds[...]
        print(f"ds[...] = {val2}, type={type(val2)}")
    except Exception as e:
        print(f"ds[...] error: {e}")
    try:
        val3 = ds[0]
        print(f"ds[0] = {val3}, type={type(val3)}")
    except Exception as e:
        print(f"ds[0] error: {e}")
    try:
        val4 = ds.value
        print(f"ds.value = {val4}, type={type(val4)}")
    except Exception as e:
        print(f"ds.value error: {e}")
    # Try reading as array
    val5 = ds[:]
    print(f"ds[:] = {val5}, type={type(val5)}")