#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 11:02:15 2023

@author: thomas
"""

# =============================================================================
# initialise
# =============================================================================
# NB run in antarctica_iom environment
## imports ##
import numpy as np
from matplotlib.path import Path
from shapely.geometry import Polygon
import scipy.spatial
import geopandas as gpd
import xarray as xr
import pyproj
import pandas as pd
import datetime

# function to compute average smb anomaly per basin between 1992 and 2017


def compute_smb(
        rxx, ryy,
        smb,
        smbt,
        basin_coords,
        t1_ref, t2_ref,
        time_resolution_flag):

    # convert basin polygon to binary mask on same grid
    points = np.vstack((rxx.flatten(), ryy.flatten())).T
    path = Path(basin_coords)
    basin_mask = path.contains_points(points)
    basin_mask = basin_mask.reshape(np.size(rxx, 0), np.size(rxx, 1))

    # check plot
    # basin_coords = np.array(basin_coords)
    # bx = basin_coords[:,0]
    # by = basin_coords[:,1]
    # import matplotlib.pyplot as plt
    # plt.figure(figsize = (10,10),constrained_layout=True),
    # ax = plt.subplot(111)
    # ax.pcolormesh(rxx, ryy, basin_mask, cmap='Greys')
    # ax.plot(bx, by)
    # ax.set_aspect('equal')

    # racmo resolution
    posting = 27e3

    ## compute smb anomaly in specified area ##
    smb_basin = []
    for epoch in range(len(smbt)):
        tmp = smb[epoch]
        # convert to Gt
        tmp = tmp*posting*posting*1e-12
        tmp = np.ma.masked_where(~basin_mask, tmp)
        smb_basin.append(tmp.sum())
        del tmp

    smb_basin = np.array(smb_basin, dtype=float)

    if time_resolution_flag == 'monthly':
        # compute annual smb (if input is monthly)
        smbt_yearly = np.unique(np.floor(smbt))
        smb_basin_yearly = np.sum(smb_basin.reshape((-1, 12)), axis=1)

    elif time_resolution_flag == 'yearly':
        smbt_yearly = smbt
        smb_basin_yearly = smb_basin

    # compute average smb within reference period
    smb_ref = smb_basin_yearly[(smbt_yearly >= t1_ref) & (
        smbt_yearly < t2_ref)].mean()

    # get anomaly
    smb_basin_yearly_anom = smb_basin_yearly - smb_ref

    smb_basin_anom_1992_2017 = smb_basin_yearly_anom[np.where(
        (smbt_yearly >= 1992) & (smbt_yearly < 2017))].mean()

    return smb_ref, smb_basin_anom_1992_2017


# =============================================================================
# load Rignot basins
# =============================================================================
basins_filein = '/Users/thomas/Documents/github/imbie_partitioning/aux/measures_basins/Basins_IMBIE_Antarctica_v02.shp'
basins_gdf = gpd.read_file(basins_filein)
basins = basins_gdf['NAME'].iloc[1:].values

# =============================================================================
# load SMB
# =============================================================================
## from racmo v2.3p2 ##
smb_file = '/Volumes/eartsl/smb_models/racmo/antarctica/smb_monthlyS_ANT27_ERA5-3H_RACMO2.3p2_197901_202112.nc'
ds = xr.open_dataset(smb_file)

smb = ds.smb.squeeze()


def year_fraction(date):
    start = datetime.date(date.year, 1, 1).toordinal()
    year_length = datetime.date(date.year+1, 1, 1).toordinal() - start
    return date.year + float(date.toordinal() - start) / year_length


smbt = [year_fraction(epoch) for epoch in pd.DatetimeIndex(ds.time)]

## convert from rotated grid ##


def ll2ps(
        lat, lon):
    """
    transform lat and lon to polarstereographic x and y coordinates referenced to 71 S

    inputs:
        lat = latitude coordinates (can be scalar, vector or matrix)
        lon = longitude coordinates (can be scalar, vector or matrix)

    outputs:
        x = polarstereographic x coordinates in m [same as lon]
        y = polarstereographic y coordinates in m [same as lat]
    """

    import numpy as np

    # set parameters
    phi_c = -71  # standard parallel
    a = 6378137.0  # radius of ellipsoid, WGS84
    e = 0.08181919  # eccentricity, WGS84
    lambda_0 = 0  # meridian along positive Y axis

    # convert to radians
    lat = np.radians(lat)
    phi_c = np.radians(phi_c)
    lon = np.radians(lon)

    t = np.tan(np.pi/4+lat/2)/((1-e*np.sin(-lat))/(1+e*np.sin(-lat)))**(e/2)

    t_c = np.tan(np.pi/4 + phi_c/2) / \
        ((1-e*np.sin(-phi_c))/(1+e*np.sin(-phi_c)))**(e/2)
    m_c = np.cos(-phi_c)/np.sqrt(1-e**2*(np.sin(-phi_c))**2)
    rho = a*m_c*t/t_c  # true scale at lat phi_c

    x = -rho*np.sin(-lon + lambda_0)
    y = rho*np.cos(-lon + lambda_0)

    return x, y


rad2deg = 180./np.pi
p = pyproj.Proj(
    '+ellps=WGS84 +proj=ob_tran +o_proj=latlon +o_lat_p=-180.0 +o_lon_p=-170.0 +lon_0=180.0')
rlon = ds.rlon.values
rlat = ds.rlat.values
x1, y1 = np.meshgrid(rlon, rlat)
lon, lat = p(x1, y1)
lon, lat = rad2deg*lon, rad2deg*lat
rxx, ryy = ll2ps(lat, lon)

t1_ref, t2_ref = 1979, 2008
del ds

# =============================================================================
# calculate smb over time period 1992-2017
# =============================================================================
print('calculating SMB using RACMOv2.3p2 27 km')

basin_smb_ref = np.full_like(basins, np.nan)
basin_smb_anom_dmdt_1992_2017 = np.full_like(basins, np.nan)

for i, basin in enumerate(basins):
    print('\n'+basin+'\n')
    region_idx = basins_gdf[basins_gdf['NAME'] ==
                            basin].index  # get basins for desired region
    if i == 12:
        basin_coords = list(
            basins_gdf['geometry'][region_idx[0]][1].exterior.coords)
    else:
        basin_coords = list(
            basins_gdf['geometry'][region_idx[0]].exterior.coords)

    basin_coords = np.array(basin_coords)

    basin_smb_ref[i], basin_smb_anom_dmdt_1992_2017[i] = compute_smb(rxx, ryy,
                                                                     smb,
                                                                     smbt,
                                                                     basin_coords,
                                                                     t1_ref, t2_ref,
                                                                     'monthly')

    del basin_coords

# save
# create array with basin names
basin_names = ["A-A", "A'-B", "B-C", "C-C'", "C'-D", "D-D'", "D'-E", "E-E'", "E'-F'",
               "F'-G", "G-H", "H-H'", "H'-I", "I-I\"", "I\"-J", "J-J\"", "J\"-K", "K'-A"]
np.savez('/Users/thomas/Documents/github/imbie_partitioning/aux/altimetry/output/basin_smb_anom_dmdt_1992_2017.npz',
         basin_smb_ref=basin_smb_ref, basin_smb_anom_dmdt_1992_2017=basin_smb_anom_dmdt_1992_2017, basin_names=basin_names)
