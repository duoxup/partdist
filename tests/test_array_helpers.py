"""Regression tests for the consolidated 1D-array helpers."""
import numpy as np
import pytest

from partdist._array_helpers import as_1d_array, as_1d_float_array


def test_accepts_1d_list():
    result = as_1d_float_array([1.0, 2.0, 3.0], "x")
    assert result.shape == (3,)
    assert result.dtype == float


def test_accepts_1d_ndarray():
    result = as_1d_array(np.array([1, 2, 3]), name="x")
    assert result.shape == (3,)


def test_rejects_2d_input():
    with pytest.raises(ValueError, match="ndim=2"):
        as_1d_float_array(np.zeros((3, 4)), "x")


def test_rejects_3d_input():
    with pytest.raises(ValueError, match="ndim=3"):
        as_1d_array(np.zeros((2, 2, 2)), name="x")


def test_dtype_coercion():
    result = as_1d_array([1, 2, 3], dtype=float, name="x")
    assert result.dtype == np.float64


def test_name_appears_in_error():
    with pytest.raises(ValueError, match="velocity"):
        as_1d_float_array(np.zeros((2, 2)), "velocity")


def test_scalar_input_becomes_length1():
    result = as_1d_float_array(3.14, "scalar")
    assert result.shape == (1,)
    assert result.dtype == float


def test_profiles_1d_rejects_2d():
    """profiles_1d.py previously silently accepted 2D input; now it must reject."""
    from partdist import profiles_1d as p1d

    # gaussian_profile calls _as_1d_float_array(x) — passing a 2D array
    # should now raise ValueError thanks to the canonical helper.
    with pytest.raises(ValueError, match="ndim=2"):
        p1d.gaussian_profile(np.zeros((3, 4)), sigma=1.0)
