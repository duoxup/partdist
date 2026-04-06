#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 01:42:32 2026

@author: duoxup
"""

import matplotlib.pyplot as plt
import partdist.pd3d.io as io
import partdist.pd3d.viz as viz

from pathlib import Path

data_dir  = Path('data')
fnames = ['original.dist', 'replicated.dist', 'multiplied.dist',
          'chirped.dist', 'matched.dist']

fig, axes = plt.subplots(ncols=len(fnames),
                         figsize=(20, 4))

for idx, fname in enumerate(fnames):
    dist = io.read_astra_distribution(data_dir / fname)
    viz.hist2d_pd3d(dist,
                    x='z', y='pz',
                    color_threshold=1e-3,
                    cmap='jet',
                    fig=fig,
                    ax=axes[idx])