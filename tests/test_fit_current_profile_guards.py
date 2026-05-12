"""Regression tests: fit_current_profile validates input shape and minimum length."""
import numpy as np
import pytest

from partdist.pd3d.analysis import fit_current_profile


def test_rejects_empty_inputs():
    with pytest.raises(ValueError, match="empty"):
        fit_current_profile(z_centers=np.array([]), I_z=np.array([]))


def test_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="same shape"):
        fit_current_profile(z_centers=np.array([0.0, 1.0]), I_z=np.array([1.0]))


def test_rejects_too_few_bins():
    """A 3-parameter profile needs at least 3 bins. Size-1 or size-2 must raise."""
    with pytest.raises(ValueError, match="at least"):
        fit_current_profile(z_centers=np.array([0.0]), I_z=np.array([1.0]))
    with pytest.raises(ValueError, match="at least"):
        fit_current_profile(z_centers=np.array([0.0, 1.0]), I_z=np.array([1.0, 0.5]))
