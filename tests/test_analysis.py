#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 01:34:39 2026

@author: duoxup
"""

from partdist.pd3d import analysis as ana

from test_io3d import dist_from_astra as dist


twiss = ana.compute_twiss_plane(dist, plane='x',
                                weight='Q_abs')