#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 17:21:29 2026

@author: duoxup
"""
import os

import partdist.pd3d.io as io
workdir = r'/lustre/fs25/group/pitz/duoxup/THz_ideal_machine/genesis/cluster00000001/case_000959/outputs/'
os.chdir(workdir)

# fname = 'Q--150pC-IPart-200000-sig_z-0.05mm.ini'
# fname = 'test1.ini'
# fname = 'ast.0528.001'
fname = 'testMatching.dist'

dist_from_astra = io.read_astra_distribution(fname)