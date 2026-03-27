#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 21:11:10 2026

@author: duoxup
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass
import warnings

from partdist.pd3d.core import ParticleDistribution

# Assuming these constants exist, replace with actual values if different
class const:
    e = 1.602176634e-19  # electron charge [C]
    m_e = 9.10938356e-31  # electron mass [kg]
    c = 299792458  # speed of light [m/s]

def match_twiss_parameters(
    dist: 'ParticleDistribution',
    beta_x: float,
    alpha_x: float,
    beta_y: float,
    alpha_y: float,
    *,
    plane: str = "both",
    reference_energy: Optional[float] = None,
    inplace: bool = False,
) -> 'ParticleDistribution':
    """
    Match transverse Twiss parameters using Courant-Snyder transformation
    
    Parameters
    ----------
    dist : ParticleDistribution
        Particle distribution object
    beta_x, alpha_x, beta_y, alpha_y : float
        Target Twiss parameters
        β: beta function [m]
        α: alpha function (dimensionless)
    plane : str, optional
        Plane to match:
        "x": match only x-plane
        "y": match only y-plane
        .32oth": match both x and y planes (default)
    reference_energy : float, optional
        Reference energy [eV] for normalized emittance calculation
        If not provided, uses mean kinetic energy of distribution
    inplace : bool, optional
        Whether to modify distribution in-place (default: False)
    
    Returns
    -------
    ParticleDistribution
        New distribution matched to target Twiss parameters
    
    Notes
    -----
    Uses Courant-Snyder transformation to match particle distribution
    to target Twiss parameters. This transformation preserves phase
    space volume (emittance).
    """
    if plane not in ("x", "y", "both"):
        raise ValueError(f"plane must be 'x', 'y', or 'both', got: {plane}")
    
    if beta_x <= 0 or beta_y <= 0:
        raise ValueError("beta functions must be positive")
    
    # Create output distribution
    out = dist if inplace else dist.copy()
    
    # Calculate reference energy if not provided
    if reference_energy is None:
        # Assuming distribution has method to calculate weighted mean
        reference_energy = out.mean("kinetic_energy_eV", weight="absQ")
    
    # Calculate relativistic parameters
    gamma = 1.0 + reference_energy * const.e / (const.m_e * const.c**2)
    beta_rel = np.sqrt(1 - 1/gamma**2)  # relativistic beta
    p0 = gamma * const.m_e * const.c * beta_rel  # Reference momentum [kg·m/s]
    
    # Helper function to apply Courant-Snyder transformation
    def apply_cs_transform(coord, coord_prime, target_beta, target_alpha):
        """
        Apply Courant-Snyder transformation to phase space coordinates
        
        Parameters
        ----------
        coord : array
            Transverse coordinate (x or y) [m]
        coord_prime : array
            Derivative dx/ds or dy/ds [rad]
        target_beta : float
            Target beta function [m]
        target_alpha : float
            Target alpha function (dimensionless)
        
        Returns
        -------
        tuple
            Transformed (coord, coord_prime)
        """
        # Calculate second moments
        sigma_11 = np.mean(coord**2)
        sigma_12 = np.mean(coord * coord_prime)
        sigma_22 = np.mean(coord_prime**2)
        
        # Calculate geometric emittance
        epsilon_geo = np.sqrt(sigma_11 * sigma_22 - sigma_12**2)
        
        if epsilon_geo <= 0:
            warnings.warn(f"Geometric emittance is non-positive: {epsilon_geo}")
            epsilon_geo = max(epsilon_geo, 1e-30)
        
        # Calculate current Twiss parameters
        beta_current = sigma_11 / epsilon_geo
        alpha_current = -sigma_12 / epsilon_geo
        # gamma_current = sigma_22 / epsilon_geo  # Not used in transformation
        
        # Courant-Snyder transformation matrix elements
        R11 = np.sqrt(target_beta / beta_current)
        R12 = 0
        R21 = (alpha_current - target_alpha) / np.sqrt(beta_current * target_beta)
        R22 = np.sqrt(beta_current / target_beta)
        
        # Apply transformation
        coord_new = R11 * coord + R12 * coord_prime
        coord_prime_new = R21 * coord + R22 * coord_prime
        
        return coord_new, coord_prime_new, epsilon_geo
    
    # Process x-plane if requested
    if plane in ("x", "both"):
        x = out.x
        # Calculate x' = dx/ds ≈ vx/vz
        vz = out.vz
        # Add small epsilon to avoid division by zero
        epsilon = 1e-30 if np.any(vz == 0) else 0
        x_prime = out.vx / (vz + epsilon)
        
        # Apply Courant-Snyder transformation
        x_new, x_prime_new, epsilon_x = apply_cs_transform(
            x, x_prime, beta_x, alpha_x
        )
        
        # Convert x' back to vx
        vx_new = x_prime_new * vz
        out.update_data('x', x_new, inplace=True)
        out.update_data('vx', vx_new, inplace=True)
        
        # Calculate normalized emittance
        epsilon_n_x = epsilon_x * gamma * beta_rel
        print(f"X-plane: geometric ε = {epsilon_x:.3e} m·rad, "
              f"normalized ε = {epsilon_n_x:.3e} m·rad")
    
    # Process y-plane if requested
    if plane in ("y", "both"):
        y = out.y
        # Calculate y' = dy/ds ≈ vy/vz
        vz = out.vz
        epsilon = 1e-30 if np.any(vz == 0) else 0
        y_prime = out.vy / (vz + epsilon)
        
        # Apply Courant-Snyder transformation
        y_new, y_prime_new, epsilon_y = apply_cs_transform(
            y, y_prime, beta_y, alpha_y
        )
        
        # Convert y' back to vy
        vy_new = y_prime_new * vz
        out.update_data('y', y_new, inplace=True)
        out.update_data('vy', vy_new, inplace=True)
        
        # Calculate normalized emittance
        epsilon_n_y = epsilon_y * gamma * beta_rel
        print(f"Y-plane: geometric ε = {epsilon_y:.3e} m·rad, "
              f"normalized ε = {epsilon_n_y:.3e} m·rad")
    
    return out


# Example usage demonstration
if __name__ == "__main__":
    # Target Twiss parameters
    beta_x_target = 0.2  # m
    alpha_x_target = 0
    beta_y_target = 0.2  # m
    alpha_y_target = -1
    
    from test_io3d import dist_from_astra as dist
    
    
    # Apply matching
    matched_dist = match_twiss_parameters(
        dist=dist,
        beta_x=beta_x_target,
        alpha_x=alpha_x_target,
        beta_y=beta_y_target,
        alpha_y=alpha_y_target,
        plane="both",
        # reference_energy=1e6,  # 1 MeV
        inplace=False
    )
    
    print("Twiss parameter matching completed successfully")
    
    
    