"""Tests for the refactored `extract_core_ellipse`:

(1) numerical equivalence — the new mask-only iteration produces the same
    survivor set as the original slice-per-iteration implementation;
(2) basic sanity — outliers are clipped, inliers retained, edge cases handled;
(3) performance regression — N=200k completes in well under the pre-fix budget.

The reference implementation is inlined here so we don't depend on git history.
"""
from __future__ import annotations

import time
import numpy as np
import pytest

from partdist import ParticleDistribution3D
from partdist.pd3d.manipulator import extract_core_ellipse


_DEFAULT_PLANES = [("x", "xp"), ("y", "yp"), ("z", "delta")]


def _reference_extract_core_ellipse(
    dist,
    n_sigma=3.0,
    *,
    planes=None,
    max_iter=20,
    weight="Q_abs",
):
    """Old slice-per-iteration algorithm, for numerical comparison."""
    from partdist.pd3d.utils import _get_weight_array
    if planes is None:
        planes = _DEFAULT_PLANES
    current = dist
    for _ in range(max_iter):
        w = _get_weight_array(current, weight, absolute=True)
        combined = np.ones(len(current.get_data("x")), dtype=bool)
        for key1, key2 in planes:
            v1 = current.get_data(key1).astype(float)
            v2 = current.get_data(key2).astype(float)
            m1 = float(np.average(v1, weights=w))
            m2 = float(np.average(v2, weights=w))
            d1 = v1 - m1
            d2 = v2 - m2
            s11 = float(np.average(d1 ** 2, weights=w))
            s12 = float(np.average(d1 * d2, weights=w))
            s22 = float(np.average(d2 ** 2, weights=w))
            det = s11 * s22 - s12 ** 2
            if det <= 0:
                continue
            maha2 = (s22 * d1 ** 2 - 2.0 * s12 * d1 * d2 + s11 * d2 ** 2) / det
            combined &= maha2 <= n_sigma ** 2
        if combined.all():
            break
        current = current.slice(combined)
    return current


def _make_dist(n=10_000, seed=7, outlier_frac=0.02):
    """Mostly Gaussian core with a small fraction of strong outliers."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1e-4, n)
    y = rng.normal(0, 1e-4, n)
    z = rng.normal(0, 1e-3, n)
    px = rng.normal(0, 1e4, n)
    py = rng.normal(0, 1e4, n)
    pz = 1e7 + rng.normal(0, 1e5, n)

    n_out = int(n * outlier_frac)
    out_idx = rng.choice(n, size=n_out, replace=False)
    x[out_idx]  += rng.normal(0, 1e-2, n_out)  # ~100σ outliers in x
    pz[out_idx] += rng.normal(0, 1e7,  n_out)  # ~100σ outliers in pz

    return ParticleDistribution3D(
        x=x, y=y, z=z,
        px=px, py=py, pz=pz,
        t=np.zeros(n), Q=np.full(n, -1.6e-19),
    )


def _key_set(dist) -> set[tuple[float, float, float, float, float, float]]:
    """Identify particles by their (x, y, z, px, py, pz) values."""
    return {
        tuple(row) for row in np.column_stack([
            dist.get_data("x"),  dist.get_data("y"),  dist.get_data("z"),
            dist.get_data("px"), dist.get_data("py"), dist.get_data("pz"),
        ])
    }


def test_extract_core_ellipse_matches_reference_implementation():
    d = _make_dist()
    fast = extract_core_ellipse(d, n_sigma=3.0)
    ref = _reference_extract_core_ellipse(d, n_sigma=3.0)
    assert len(fast) == len(ref)
    assert _key_set(fast) == _key_set(ref)


def test_extract_core_ellipse_clips_strong_outliers():
    d = _make_dist(n=5_000, outlier_frac=0.04)
    n_before = len(d)
    core = extract_core_ellipse(d, n_sigma=3.0)
    assert len(core) < n_before, "Some outliers should have been clipped"
    assert len(core) > 0.85 * n_before, "Core should retain bulk of particles"


def test_extract_core_ellipse_returns_input_when_no_outliers():
    """A clean Gaussian beam: first-iteration covariance is already self-consistent
    so almost nothing is clipped (within Mahalanobis 3σ ≈ 0.4% expected loss)."""
    rng = np.random.default_rng(0)
    n = 5_000
    d = ParticleDistribution3D(
        x=rng.normal(0, 1e-4, n), y=rng.normal(0, 1e-4, n), z=rng.normal(0, 1e-3, n),
        px=rng.normal(0, 1e4, n), py=rng.normal(0, 1e4, n),
        pz=1e7 + rng.normal(0, 1e5, n),
        t=np.zeros(n), Q=np.full(n, -1.6e-19),
    )
    core = extract_core_ellipse(d, n_sigma=4.0)  # 4σ — keep ≥99% of a clean Gaussian
    assert len(core) >= 0.97 * n


def test_extract_core_ellipse_rejects_bad_n_sigma():
    d = _make_dist(n=100)
    with pytest.raises(ValueError):
        extract_core_ellipse(d, n_sigma=0.0)
    with pytest.raises(ValueError):
        extract_core_ellipse(d, n_sigma=-1.0)


def test_extract_core_ellipse_rejects_bad_max_iter():
    d = _make_dist(n=100)
    with pytest.raises(ValueError):
        extract_core_ellipse(d, max_iter=0)


def test_extract_core_ellipse_perf_under_budget():
    """At N=2e5 with 2% outliers the fast path should finish well under 2s
    (pre-fix this was ~6-12s on the same machine, slice-bound)."""
    d = _make_dist(n=200_000, outlier_frac=0.02)
    t0 = time.perf_counter()
    core = extract_core_ellipse(d, n_sigma=3.0)
    elapsed = time.perf_counter() - t0
    assert len(core) < len(d)
    assert elapsed < 2.0, f"extract_core_ellipse N=2e5 took {elapsed:.2f}s, budget 2s"
