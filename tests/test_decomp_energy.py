#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 22:57:28 2026

@author: duoxup
"""

from test_io3d import dist_from_astra as dist
import matplotlib.pyplot as plt

import partdist.pd3d.manipulator as manip

z_array, energy_array = dist.z, dist.kinetic_energy_eV
# Assuming you have z_array and energy_array
result = manip.diagnose_and_decompose_energy(z_array, energy_array, smoothing_factor=None)

# Print key diagnostics
print("Non-correlated spread std (to preserve):", result['diagnostics']['uncorrelated_spread_eV'], "eV")
print("Correlated spread std (to replace):", result['diagnostics']['correlated_spread_eV'], "eV")

# Visualize the decomposition
fig, axs = manip.plot_decomposition(result)
plt.show()