#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 16:29:29 2026

@author: duoxup
"""
import numpy as np

from partdist.pd3d.core import ParticleDistribution

x = np.array([0.0, 1e-6, -1e-6])
y = np.array([0.0, 2e-6, -2e-6])
z = np.array([0.0, 1e-3, -1e-3])

vx = np.array([0.0, 1e5, -1e5])
vy = np.array([0.0, 2e5, -2e5])
vz = np.array([2.9e8, 2.9e8, 2.9e8])

t = np.array([0.0, 3e-12, -3e-12])
Q = np.array([-1e-12, -1e-12, -1e-12])
pid = np.array([10, 11, 12], dtype=np.int64)

dist = ParticleDistribution.from_arrays(
    x=x,
    y=y,
    z=z,
    vx=vx,
    vy=vy,
    vz=vz,
    t=t,
    Q=Q,
    extras={"pid": pid},
)

print(dist.particle_quantity_keys)
# print(dist.statistical_quantity_keys[:5])

print(dist.mean("z"))
print(dist.std("z"))
print(dist.pz)
print(dist.kinetic_energy_eV)
print(dist.id)
# print(dist.quantity_kind("x"))
# print(dist.quantity_kind("mean_x"))

paq = dist.get_quantity('x')