#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 17:21:29 2026

@author: duoxup
"""
import os

from partdist.pd3d.io import read_astra_distribution, write_astra_distribution

workdir = r'/lustre/fs25/group/pitz/duoxup/THzSuperRad/debug/Q=-150pC-beta_x_scale_from_opt=1.000/'
os.chdir(workdir)

fname = 'Q--150pC-IPart-200000-sig_z-0.05mm.ini'
# fname = 'test1.ini'
# fname = 'ast.0528.001'

dist_from_astra = read_astra_distribution(fname)