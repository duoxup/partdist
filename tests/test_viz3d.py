#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 17:46:17 2026

@author: duoxup
"""

import xtils

from test_io3d import dist_from_astra
from partdist.pd3d.viz import (scatter_pd3d, 
                               hist2d_pd3d, 
                               # _add_projection_curves_pd3d
                               )

fig, axes = xtils.new_subplots(2, base_size=(4, 3),
                               layout='constrained')

scatter_pd3d(dist_from_astra, x='z', y='pz', fig=fig, ax=axes[0])
hist2d_pd3d(dist_from_astra, x='z', y='pz', weight='Q_abs',
            fig=fig, ax=axes[1],
            cmap='jet', color_threshold=1e-2)

# add_projection_curves_pd3d(dist_from_astra, ax=axes[1], x='z', y='pz',
#                            normalize=True,
#                            xproj_scale=0.1,
#                            yproj_scale=0.1,
#                            show_xproj_axis=True,
#                            show_yproj_axis=True,
#                            )