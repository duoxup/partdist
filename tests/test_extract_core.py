"""
Visualise the effect of each extract_core method on the longitudinal phase
space (z – delta).

Run:
    python tests/test_extract_core.py
Output:
    tests/test_extract_core.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import partdist.pd3d.io as io
from partdist.pd3d.manipulator import (
    extract_core_sigma_clip,
    extract_core_percentile,
    extract_core_current_threshold,
    extract_core_ellipse,
    extract_core_profile_fit,
)
from partdist.pd3d.viz import hist2d_pd3d

DIST_FILE = Path.home() / "simdata/pitz/S2E_ideal_machine/case1/cmp.dist"
OUT_FILE  = Path(__file__).parent / "test_extract_core.png"

CASES = [
    (None,                       {},                                                   "original"),
    (extract_core_sigma_clip,    dict(n_sigma=3.0),                                   "sigma_clip\nn_sigma=3"),
    (extract_core_sigma_clip,    dict(n_sigma=2.0),                                   "sigma_clip\nn_sigma=2"),
    (extract_core_percentile,    dict(percentile=90),                                 "percentile\n90 %"),
    (extract_core_percentile,    dict(percentile=70),                                 "percentile\n70 %"),
    (extract_core_current_threshold, dict(threshold=0.1),                             "current_threshold\nthreshold=0.10"),
    (extract_core_current_threshold, dict(threshold=0.3),                             "current_threshold\nthreshold=0.30"),
    (extract_core_ellipse,       dict(n_sigma=3.0),                                   "ellipse\nn_sigma=3"),
    (extract_core_ellipse,       dict(n_sigma=2.5, planes=[("z", "delta")]),          "ellipse (z–δ only)\nn_sigma=2.5"),
    (extract_core_profile_fit,   dict(n_sigma=2.0, profile="gaussian"),               "profile_fit gaussian\nn_sigma=2"),
    (extract_core_profile_fit,   dict(n_sigma=2.0, profile="parabola"),               "profile_fit parabola\nn_sigma=2"),
    (extract_core_profile_fit,   dict(n_sigma=1.5, profile="gaussian"),               "profile_fit gaussian\nn_sigma=1.5"),
]

N_COLS = 3
N_ROWS = -(-len(CASES) // N_COLS)   # ceiling division


def main():
    print(f"Reading {DIST_FILE} ...")
    dist_all = io.read_astra_distribution(DIST_FILE)

    # filter alive particles (status >= 0)
    if "status" in dist_all.extra_quantity_keys:
        import numpy as np
        mask = dist_all.get_data("status").astype(int) >= 0
        dist = dist_all.slice(mask)
    else:
        dist = dist_all

    n_orig = len(dist.get_data("z"))
    print(f"  alive particles: {n_orig}")

    fig, axes = plt.subplots(N_ROWS, N_COLS,
                             figsize=(N_COLS * 5, N_ROWS * 4))
    axes = axes.flatten()

    for ax, (func, kw, label) in zip(axes, CASES):
        if func is None:
            d = dist
        else:
            d = func(dist, **kw)

        n = len(d.get_data("z"))
        ratio = n / n_orig * 100

        hist2d_pd3d(
            d,
            x="z", y="delta",
            fig=fig, ax=ax,
            color_threshold=0.01,
            cmap="jet",
            colorbar=False,
        )
        ax.set_title(f"{label}\nN={n} ({ratio:.1f}%)", fontsize=9)

    # hide unused axes
    for ax in axes[len(CASES):]:
        ax.set_visible(False)

    fig.suptitle(f"extract_core — z–δ phase space\n{DIST_FILE.name}", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_FILE, dpi=150)
    print(f"Saved: {OUT_FILE}")


if __name__ == "__main__":
    main()
