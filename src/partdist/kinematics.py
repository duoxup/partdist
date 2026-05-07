"""
Particle-agnostic conversions among common relativistic kinematic quantities:

    - gamma  (Lorentz factor)
    - beta   (v/c)
    - v      (speed, m/s)
    - ke_eV  (kinetic energy, eV)
    - p_eVc  (momentum, eV/c)
    - p_norm (= p / (m0 * c), dimensionless "normalized momentum")

Design goals
------------
- Minimal, readable API.
- Particle-agnostic: every function accepts rest mass ``m0`` [kg] and
  charge ``q`` [C] (the sign of ``q`` does not matter — magnitude is
  used internally).
- Robust scalar/array behaviour: scalars in -> scalar out; arrays/lists
  in -> ``numpy`` arrays out.
- Pairwise conversions: call a specific ``X_from_Y`` function or use
  the generic :func:`convert_motion`.

Physical constants are taken from :mod:`scipy.constants` (electron
defaults: ``m0 = m_e``, ``q = e``, ``c = c``).
"""

from __future__ import annotations

import numpy as np
from scipy.constants import c as g_c, m_e as g_m0, e as g_e0


# ---------------------------------------------------------------------
# Internal helpers (keep private and minimal)
# ---------------------------------------------------------------------
def _as_array(x):
    """Return (arr, is_scalar). Scalars remain scalar on output via _maybe_scalar."""
    arr = np.asarray(x, dtype=float)
    is_scalar = np.isscalar(x) or arr.ndim == 0
    return arr, is_scalar

def _maybe_scalar(arr, is_scalar):
    """Return float if input was scalar; otherwise return ndarray."""
    if is_scalar:
        return float(np.asarray(arr).reshape(1)[0])
    return np.asarray(arr)

def _absq(q: float) -> float:
    """Return absolute elementary charge to handle negative electron charge cleanly."""
    return float(abs(q))


# ---------------------------------------------------------------------
# Canonical pivot relations (gamma <-> each quantity)
# Keep these small and mathematically explicit.
# ---------------------------------------------------------------------
def gamma_from_beta(beta):
    beta, s = _as_array(beta)
    if np.any((beta < 0) | (beta >= 1)):
        raise ValueError("beta must satisfy 0 <= beta < 1.")
    out = 1.0 / np.sqrt(1.0 - beta**2)
    return _maybe_scalar(out, s)

def beta_from_gamma(gamma):
    gamma, s = _as_array(gamma)
    if np.any(gamma < 1.0):
        raise ValueError("gamma must be >= 1.")
    out = np.sqrt(1.0 - 1.0 / (gamma**2))
    return _maybe_scalar(out, s)

def gamma_from_v(v, *, c: float = g_c):
    v, s = _as_array(v)
    if np.any((v < 0) | (v >= c)):
        raise ValueError("speed v must satisfy 0 <= v < c.")
    beta = v / c
    out = gamma_from_beta(beta)
    return _maybe_scalar(out, s)

def v_from_gamma(gamma, *, c: float = g_c):
    beta = beta_from_gamma(gamma)
    # beta_from_gamma preserves scalar/array; convert to array to multiply safely
    beta_arr, s = _as_array(beta)
    out = beta_arr * c
    return _maybe_scalar(out, s)

def gamma_from_ke_eV(ke_eV, *, m0: float = g_m0, q: float = g_e0, c: float = g_c):
    """gamma = 1 + (KE / m0 c^2);  KE in eV (uses |q| to convert eV->J)."""
    ke_eV, s = _as_array(ke_eV)
    if np.any(ke_eV < 0):
        raise ValueError("Kinetic energy (eV) must be >= 0.")
    m0c2_J = m0 * c**2
    ke_J   = _absq(q) * ke_eV
    out = 1.0 + ke_J / m0c2_J
    return _maybe_scalar(out, s)

def ke_eV_from_gamma(gamma, *, m0: float = g_m0, q: float = g_e0, c: float = g_c):
    """KE[eV] = (gamma - 1) m0 c^2 / |q|."""
    gamma, s = _as_array(gamma)
    if np.any(gamma < 1.0):
        raise ValueError("gamma must be >= 1.")
    m0c2_J = m0 * c**2
    ke_J = (gamma - 1.0) * m0c2_J
    out = ke_J / _absq(q)
    return _maybe_scalar(out, s)

def gamma_from_p_eVc(p_eVc, *, m0: float = g_m0, q: float = g_e0, c: float = g_c):
    """
    gamma = sqrt(1 + (p/m0c)^2); input p in eV/c, i.e. energy units divided by c.
    Convert p[eV/c] -> p[J*s/m] by multiplying by |q| (since 1 eV = |q| J).
    But more simply: p_eVc / (m0 c^2 / |q|) == (p/m0c) in dimensionless form.
    """
    p_eVc, s = _as_array(p_eVc)
    m0c2_eV = (m0 * c**2) / _absq(q)  # rest energy in eV
    out = np.sqrt(1.0 + (p_eVc / m0c2_eV)**2)
    return _maybe_scalar(out, s)

def p_eVc_from_gamma(gamma, *, m0: float = g_m0, q: float = g_e0, c: float = g_c):
    """
    p[eV/c] = m0 c * sqrt(gamma^2 - 1) converted to eV/c scale:
    p_eVc = (m0 c^2 / |q|) * sqrt(gamma^2 - 1)  == m0c2_eV * sqrt(gamma^2 - 1).
    """
    gamma, s = _as_array(gamma)
    if np.any(gamma < 1.0):
        raise ValueError("gamma must be >= 1.")
    m0c2_eV = (m0 * c**2) / _absq(q)
    out = m0c2_eV * np.sqrt(gamma**2 - 1.0)
    return _maybe_scalar(out, s)

def gamma_from_p_norm(p_norm):
    """
    p_norm = p / (m0 c)  (dimensionless); valid for any particle.
    gamma = sqrt(1 + p_norm^2)
    """
    p_norm, s = _as_array(p_norm)
    if np.any(p_norm < 0):
        raise ValueError("Normalized momentum must be >= 0.")
    out = np.sqrt(1.0 + p_norm**2)
    return _maybe_scalar(out, s)

def p_norm_from_gamma(gamma):
    gamma, s = _as_array(gamma)
    if np.any(gamma < 1.0):
        raise ValueError("gamma must be >= 1.")
    out = np.sqrt(gamma**2 - 1.0)
    return _maybe_scalar(out, s)


# ---------------------------------------------------------------------
# Cross conversions via gamma (thin wrappers for convenience)
# Each pairwise conversion is implemented by composing pivot relations.
# ---------------------------------------------------------------------
# beta <-> v
def beta_from_v(v, *, c: float = g_c):
    v, s = _as_array(v)
    if np.any((v < 0) | (v >= c)):
        raise ValueError("speed v must satisfy 0 <= v < c.")
    out = v / c
    return _maybe_scalar(out, s)

def v_from_beta(beta, *, c: float = g_c):
    beta, s = _as_array(beta)
    if np.any((beta < 0) | (beta >= 1)):
        raise ValueError("beta must satisfy 0 <= beta < 1.")
    out = beta * c
    return _maybe_scalar(out, s)

# beta <-> ke_eV
def beta_from_ke_eV(ke_eV, *, m0: float = g_m0, q: float = g_e0, c: float = g_c):
    g = gamma_from_ke_eV(ke_eV, m0=m0, q=q, c=c)
    return beta_from_gamma(g)

def ke_eV_from_beta(beta, *, m0: float = g_m0, q: float = g_e0, c: float = g_c):
    g = gamma_from_beta(beta)
    return ke_eV_from_gamma(g, m0=m0, q=q, c=c)

# beta <-> p_eVc
def beta_from_p_eVc(p_eVc, *, m0: float = g_m0, q: float = g_e0, c: float = g_c):
    g = gamma_from_p_eVc(p_eVc, m0=m0, q=q, c=c)
    return beta_from_gamma(g)

def p_eVc_from_beta(beta, *, m0: float = g_m0, q: float = g_e0, c: float = g_c):
    g = gamma_from_beta(beta)
    return p_eVc_from_gamma(g, m0=m0, q=q, c=c)

# beta <-> p_norm
def beta_from_p_norm(p_norm):
    g = gamma_from_p_norm(p_norm)
    return beta_from_gamma(g)

def p_norm_from_beta(beta):
    g = gamma_from_beta(beta)
    return p_norm_from_gamma(g)

# v <-> ke_eV
def v_from_ke_eV(ke_eV, *, m0: float = g_m0, q: float = g_e0, c: float = g_c):
    return v_from_gamma(gamma_from_ke_eV(ke_eV, m0=m0, q=q, c=c), c=c)

def ke_eV_from_v(v, *, m0: float = g_m0, q: float = g_e0, c: float = g_c):
    return ke_eV_from_gamma(gamma_from_v(v, c=c), m0=m0, q=q, c=c)

# v <-> p_eVc
def v_from_p_eVc(p_eVc, *, m0: float = g_m0, q: float = g_e0, c: float = g_c):
    return v_from_gamma(gamma_from_p_eVc(p_eVc, m0=m0, q=q, c=c), c=c)

def p_eVc_from_v(v, *, m0: float = g_m0, q: float = g_e0, c: float = g_c):
    return p_eVc_from_gamma(gamma_from_v(v, c=c), m0=m0, q=q, c=c)

# v <-> p_norm
def v_from_p_norm(p_norm, *, c: float = g_c):
    return v_from_gamma(gamma_from_p_norm(p_norm), c=c)

def p_norm_from_v(v, *, c: float = g_c):
    return p_norm_from_gamma(gamma_from_v(v, c=c))

# ke_eV <-> p_eVc
def p_eVc_from_ke_eV(ke_eV, *, m0: float = g_m0, q: float = g_e0, c: float = g_c):
    return p_eVc_from_gamma(gamma_from_ke_eV(ke_eV, m0=m0, q=q, c=c), m0=m0, q=q, c=c)

def ke_eV_from_p_eVc(p_eVc, *, m0: float = g_m0, q: float = g_e0, c: float = g_c):
    return ke_eV_from_gamma(gamma_from_p_eVc(p_eVc, m0=m0, q=q, c=c), m0=m0, q=q, c=c)

# p_norm <-> (others)
def ke_eV_from_p_norm(p_norm, *, m0: float = g_m0, q: float = g_e0, c: float = g_c):
    return ke_eV_from_gamma(gamma_from_p_norm(p_norm), m0=m0, q=q, c=c)

def p_eVc_from_p_norm(p_norm, *, m0: float = g_m0, q: float = g_e0, c: float = g_c):
    return p_eVc_from_gamma(gamma_from_p_norm(p_norm), m0=m0, q=q, c=c)

def p_norm_from_ke_eV(ke_eV, *, m0: float = g_m0, q: float = g_e0, c: float = g_c):
    return p_norm_from_gamma(gamma_from_ke_eV(ke_eV, m0=m0, q=q, c=c))

def p_norm_from_p_eVc(p_eVc, *, m0: float = g_m0, q: float = g_e0, c: float = g_c):
    return p_norm_from_gamma(gamma_from_p_eVc(p_eVc, m0=m0, q=q, c=c))


# ---------------------------------------------------------------------
# Generic pairwise conversion (with robust key normalization)
# ---------------------------------------------------------------------
# Accept common spellings; keys are compared in lowercase.
ALIASES = {
    "gamma":   "gamma",
    "beta":    "beta",
    "v":       "v",
    "speed":   "v",

    # kinetic energy in eV
    "ke":      "ke_ev",
    "ke_ev":   "ke_ev",
    "t_ev":    "ke_ev",
    "kinetic": "ke_ev",
    "kinetic_energy": "ke_ev",

    # momentum in eV/c
    "p_evc":   "p_evc",
    "pc":      "p_evc",
    "p_over_c":"p_evc",
    "p":       "p_evc",

    # normalized momentum p/(m0 c)
    "p_norm":  "p_norm",
    "pnorm":   "p_norm",
}

_SUPPORTED = set(ALIASES.keys())  # acceptable input spellings

def _canon(name: str) -> str:
    """Normalize a quantity name to canonical routing keys: gamma, beta, v, ke_ev, p_evc, p_norm."""
    key = name.strip().lower()
    if key not in ALIASES:
        raise ValueError(f"Unknown quantity '{name}'. Allowed: {sorted(_SUPPORTED)}")
    return ALIASES[key]

def convert_motion(value,
                   source: str,
                   target: str,
                   *,
                   m0: float = g_m0,
                   q: float = g_e0,
                   c: float = g_c):
    """
    Convert 'value' from 'source' representation to 'target' representation.
    Supported spellings (case-insensitive): {sorted(_SUPPORTED)}.

    Canonical targets used internally: gamma, beta, v, ke_ev, p_evc, p_norm.
    """
    s = _canon(source)
    t = _canon(target)

    # source -> gamma
    if s == "gamma":
        g = value
    elif s == "beta":
        g = gamma_from_beta(value)
    elif s == "v":
        g = gamma_from_v(value, c=c)
    elif s == "ke_ev":
        g = gamma_from_ke_eV(value, m0=m0, q=q, c=c)
    elif s == "p_evc":
        g = gamma_from_p_eVc(value, m0=m0, q=q, c=c)
    elif s == "p_norm":
        g = gamma_from_p_norm(value)
    else:
        raise AssertionError("unreachable (source)")

    # gamma -> target
    if   t == "gamma":  return g
    elif t == "beta":   return beta_from_gamma(g)
    elif t == "v":      return v_from_gamma(g, c=c)
    elif t == "ke_ev":  return ke_eV_from_gamma(g, m0=m0, q=q, c=c)
    elif t == "p_evc":  return p_eVc_from_gamma(g, m0=m0, q=q, c=c)
    elif t == "p_norm": return p_norm_from_gamma(g)
    else:
        raise AssertionError("unreachable (target)")
