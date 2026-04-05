#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 20:46:56 2026

@author: duoxup
"""

# from test_io3d import dist_from_astra as dist

import partdist.pd3d.manipulator as manip
import partdist.pd3d.analysis as ana
import partdist.pd3d.viz as viz
import partdist.pd3d.io as io


def _inverted_parabola(z_max):
    return lambda z: 3 / (4*z_max) * (1 - z**2 / z_max**2)


fname = r'/lustre/fs25/group/pitz/duoxup/THzSuperRad/debug/Q=-150pC-beta_x_scale_from_opt=1.000/Q--150pC-IPart-200000-sig_z-0.05mm.ini'
dist = io.read_astra_distribution(fname)

dist = manip.replicate_longitudinally(dist, 5, 3e-4, sort_by=None)




dist_current_rescaled = manip.multiply_longitudinal_profile(dist, _inverted_parabola(9e-4),
                                                            center='mean',
                                                            normalize=True)
# viz.hist2d_pd3d(dist_current_rescaled,
#                 x='z', y='pz',
#                 color_threshold=1e-2,
#                 cmap='jet')


cor_Ekin_tar = -200e3 #-200keV
chirp_rate_tar = cor_Ekin_tar/dist_current_rescaled.std('z')
dist_chirped = manip.set_linear_chirp(dist_current_rescaled,
                                      slope=chirp_rate_tar,
                                      center_x=True,
                                      center_y=True,
                                      preserve_mean_kinetic_energy=True,
                                      weight_for_energy="Q",
                                      )

# viz.hist2d_pd3d(dist_chirped,
#                 x='z', y='pz',
#                 color_threshold=1e-2,
#                 cmap='jet')
#%%

import numpy as np
from xtils import relconv

vabs = np.sqrt(dist_chirped.vx**2 + dist_chirped.vy**2 + dist_chirped.vz**2)
ke_from_v = relconv.ke_eV_from_v(vabs)

mean_ke_from_v = np.average(ke_from_v, weights=np.abs(dist_chirped.Q))
mean_ke_builtin = dist_chirped.mean("kinetic_energy_eV", weight="absQ")

print("mean_ke_from_v   =", mean_ke_from_v)
print("mean_ke_builtin  =", mean_ke_builtin)
#%%
dist_retwissed = manip.match_twiss_xy(dist_chirped, 
                                      alpha_x = 0,
                                      beta_x = 0.2,
                                      alpha_y = -1,
                                      beta_y = 0.2,
                                      weight='Q_abs',
                                      center_before_match=True,
                                      preserve_centroid=True,
                                      )
# dist_retwissed = manip.match_twiss_y(dist, -1, 0.2, weight='Q_abs')

# print(dist.twiss)
# print(ana.compute_twiss_plane(dist_retwissed, plane='x', weight='Q_abs').alpha)
# print(ana.compute_twiss_plane(dist_retwissed, plane='x', weight='Q_abs').beta)
# print(ana.compute_twiss_plane(dist_retwissed, plane='y', weight='Q_abs').alpha)
# print(ana.compute_twiss_plane(dist_retwissed, plane='y', weight='Q_abs').beta)

# # viz.hist2d_pd3d(dist_retwissed,
# #                 x='z', y='pz',
# #                 color_threshold=1e-3,
# #                 cmap='jet')


# io.write_astra_distribution('test_manip3d.ini', dist_retwissed)


#%%
import numpy as np
from xtils import relconv

px = dist_chirped.px
py = dist_chirped.py
pz = dist_chirped.pz
pabs = np.sqrt(px**2 + py**2 + pz**2)
ke_manual = relconv.ke_eV_from_p_eVc(pabs)

mean_ke_manual = np.average(ke_manual, weights=np.abs(dist_chirped.Q))
mean_ke_builtin = dist_chirped.mean("kinetic_energy_eV", weight="absQ")

print(mean_ke_manual)
print(mean_ke_builtin)





