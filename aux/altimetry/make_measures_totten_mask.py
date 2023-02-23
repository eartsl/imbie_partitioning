#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: Lin Gilbert (UCL/MSSL/CPOM)
Date: 05/04/2022
Copyright: UCL/MSSL/CPOM. Not to be used outside UCL/MSSL/CPOM without permission of author
"""

# NB must run in cpom conda environment

import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from matplotlib.path import Path
import shapefile
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import os
os.chdir("/Users/thomas/Documents/github/cpom_software")
from cpom.gridding.gridareas import GridArea
from cpom.areas.areas import Area
# Set up main directory, for input and output files

maindir = '/Users/thomas/Documents/github/imbie_partitioning/aux/altimetry/output/'

# Read in the boundary shape file

sf = shapefile.Reader(
    '/Users/thomas/Documents/github/imbie_partitioning/aux/measures_basins/Basins_IMBIE_Antarctica_v02.shp')
shapes = sf.shapes()

# Make the mask by selecting where grid cell
# centres are within the polygon. Start by setting up the grid, then go from there.

ant_area = Area('antarctica_all')
ant_grid = GridArea('antarctica', 5e3)

measures_mask = np.zeros(ant_grid.xmesh.shape)

grid_pts = np.transpose(
    np.array([ant_grid.xmesh.flatten(), ant_grid.ymesh.flatten()]))

for i, basin in enumerate(shapes):
    if i > 0:
        this_shape = np.array(basin.points)
        this_poly = Polygon(this_shape)
        this_path = Path(this_poly.boundary)
        inside_points = this_path.contains_points(grid_pts)
        q = np.where(inside_points.reshape(ant_grid.xmesh.shape))
        measures_mask[q] = i

# Plot and save

plt.imshow(measures_mask, interpolation='none', origin='lower')
plt.show()

# create array with basin names
basin_names = ["A-A", "A'-B", "B-C", "C-C'", "C'-D", "D-D'", "D'-E", "E-E'", "E'-F'",
               "F'-G", "G-H", "H-H'", "H'-I", "I-I\"", "I\"-J", "J-J\"", "J\"-K", "K'-A"]

np.savez(maindir+'measures_mask.npz',
         measures_mask=measures_mask, basin_names=basin_names)
