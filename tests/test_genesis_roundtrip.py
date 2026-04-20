"""
Roundtrip test: read Astra dist → write Genesis h5 → read back → compare stats.
"""
import tempfile
from pathlib import Path

import numpy as np
import partdist.pd3d.io as io

# ── paths ──────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data"
ASTRA_FILE = DATA_DIR / "gen_40MeV.dist"

# ── parameters ─────────────────────────────────────────────────────────────
LAMBDA0  = 3e-4      # m  (resonant wavelength)
NPART    = 4096      # particles per slice
SAMPLE   = 1

# ── helpers ────────────────────────────────────────────────────────────────

def weighted_stats(dist, key):
    """Return (mean, std) weighted by Q_abs."""
    x = dist.get_data(key).astype(float)
    w = np.abs(dist.get_data("Q").astype(float))
    mean = float(np.average(x, weights=w))
    var  = float(np.average((x - mean)**2, weights=w))
    return mean, float(np.sqrt(var))


def print_row(label, val_orig, val_rt, unit="", rtol=0.05, atol=0.0):
    ok = abs(val_rt - val_orig) <= rtol * abs(val_orig) + atol
    flag = "OK" if ok else "FAIL"
    print(f"  {label:<25s}  {val_orig:+.4e}  {val_rt:+.4e}  {unit:<8s}  [{flag}]")


# ── main ───────────────────────────────────────────────────────────────────

def main():
    # 1. Read original Astra distribution
    orig = io.read_astra_distribution(ASTRA_FILE)
    z    = orig.get_data("z").astype(float)
    Q    = orig.get_data("Q").astype(float)
    w    = np.abs(Q)

    z_mean = float(np.average(z, weights=w))
    z_std  = float(np.sqrt(np.average((z - z_mean)**2, weights=w)))
    Q_tot  = float(w.sum())

    # time window: centre ±3σ, padded to full bunch + small margin
    margin = 0.5 * LAMBDA0
    s0   = z.min() - margin
    slen = (z.max() - z.min()) + 2 * margin

    print(f"Input:  N={len(z)}, Q_tot={Q_tot:.3e} C")
    print(f"        z ∈ [{z.min():.3e}, {z.max():.3e}] m  (σ_z={z_std:.3e} m)")
    print(f"Window: s0={s0:.3e} m,  slen={slen:.3e} m")
    print(f"        lambda0={LAMBDA0:.3e} m,  sample={SAMPLE}")
    print()

    # 2. Write Genesis h5
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        h5_path = Path(tmp.name)

    io.write_genesis_distribution(
        orig,
        h5_path,
        lambda0=LAMBDA0,
        s0=s0,
        slen=slen,
        npart=NPART,
        sample=SAMPLE,
        theta_method="direct",
        smooth_current=False,
        on_warning="warn",
        seed=42,
    )
    print(f"Written: {h5_path}")

    # 3. Read back
    rt = io.read_genesis_distribution(h5_path, drop_zero_charge=True)
    z_rt = rt.get_data("z").astype(float)
    Q_rt = rt.get_data("Q").astype(float)
    w_rt = np.abs(Q_rt)
    print(f"Readback: N={len(z_rt)}, Q_tot={w_rt.sum():.3e} C")
    print()

    # 4. Compare statistics
    print(f"{'Quantity':<25s}  {'original':>12s}  {'roundtrip':>12s}  {'unit':<8s}")
    print("  " + "-" * 70)

    # Total charge
    print_row("Q_total",  Q_tot, float(w_rt.sum()), "C", rtol=0.02)

    # Centroid and RMS for each phase-space coordinate
    for key, unit in [("x", "m"), ("y", "m"), ("z", "m"),
                      ("xp", "rad"), ("yp", "rad")]:
        try:
            m_o, s_o = weighted_stats(orig, key)
            m_r, s_r = weighted_stats(rt,   key)
            # centroid tolerance: 1% of σ (original may be ~0 due to centering)
            print_row(f"<{key}>",  m_o, m_r, unit, rtol=0.05, atol=0.01 * s_o)
            print_row(f"σ_{key}",  s_o, s_r, unit, rtol=0.05)
        except Exception as e:
            print(f"  {key}: skipped ({e})")

    # Mean kinetic energy and energy spread
    try:
        m_o, s_o = weighted_stats(orig, "kinetic_energy_eV")
        m_r, s_r = weighted_stats(rt,   "kinetic_energy_eV")
        print_row("<E_kin>",   m_o, m_r, "eV", rtol=0.01)
        print_row("σ_Ekin",    s_o, s_r, "eV", rtol=0.05)
    except Exception as e:
        print(f"  kinetic_energy: skipped ({e})")

    # Normalised transverse emittances
    try:
        from partdist.pd3d.core import ParticleDistribution3D
        if isinstance(orig, ParticleDistribution3D):
            print_row("emit_x", orig.emit_x, rt.emit_x, "m·rad", rtol=0.10)
            print_row("emit_y", orig.emit_y, rt.emit_y, "m·rad", rtol=0.10)
    except Exception as e:
        print(f"  emittances: skipped ({e})")

    print()
    h5_path.unlink(missing_ok=True)
    print("Temp file removed.")


if __name__ == "__main__":
    main()
