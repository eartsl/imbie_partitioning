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
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import pandas as pd
import numpy as np
import os
import sys
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from scipy import interpolate
import scipy.stats
import geopandas as gpd
from tabulate import tabulate

sys.dont_write_bytecode = True
sys.path.append('/Users/thomas/Documents/github/imbie_partitioning/main')

os.chdir('/Users/thomas/Documents/github/imbie_partitioning/main')

# =============================================================================
# 1. load data
# =============================================================================
# CPOM RA in Rignot basins

ra_cpom = np.load(
    '/Users/thomas/Documents/github/imbie_partitioning/aux/altimetry/output/altimetry_dm_measures_basins.npz')

time_ra_cpom = ra_cpom['t']
dm_ra_cpom = ra_cpom['dm_snowice']

# Rignot discharge aggregated into basins
discharge_rignot = pd.read_excel(
    '/Users/thomas/Documents/github/imbie_partitioning/aux/iom_datasets/pnas.1812883116.sd01_basins_discharge_1992_2017.xlsx', sheet_name='basins_discharge_1992_2017')
discharge_rignot_basins = discharge_rignot['Average discharge (Gt/yr)'].values

# load Rignot basins
basins_filein = '/Users/thomas/Documents/github/imbie_partitioning/aux/measures_basins/Basins_IMBIE_Antarctica_v02.shp'
basins_gdf = gpd.read_file(basins_filein)
basins = basins_gdf['NAME'].iloc[1:].values

# load antarctic grounding/coastline for plotting
coast_filein = '/Volumes/eartsl/MEaSUREs/Antarctic_Boundaries_from_Satellite_Radar_V2/Coastline_Antarctica_v02.shp'
coast_gdf = gpd.read_file(coast_filein)
coast_gdf['points'] = coast_gdf.apply(
    lambda x: [y for y in x['geometry'].exterior.coords], axis=1)
coast_coords = list(coast_gdf['points'][0])
cx = [i[0] for i in coast_coords]
cy = [i[1] for i in coast_coords]
del coast_gdf, coast_coords

gl_filein = '/Volumes/eartsl/MEaSUREs/Antarctic_Boundaries_from_Satellite_Radar_V2/GroundingLine_Antarctica_v02.shp'
gl_gdf = gpd.read_file(gl_filein)
gl_coords = list(gl_gdf["geometry"])
gl_coords = gl_coords[0]
del gl_gdf

# =============================================================================
# 2. find altimetry dM/dt in overlap period
# =============================================================================
# apply linear regression to each row to get dM/dt
dmdt_ra_cpom = np.full((np.shape(dm_ra_cpom)[0]), np.nan)
for i, row in enumerate(dm_ra_cpom):
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(
        time_ra_cpom[np.where(time_ra_cpom < 2017)],
        row[np.where(time_ra_cpom < 2017)])
    dmdt_ra_cpom[i] = slope

# =============================================================================
# 3. Partition into dynamics
# =============================================================================
# load smb anomaly dmdt during overlap period
basins_smb = np.load(
    '/Users/thomas/Documents/github/imbie_partitioning/aux/altimetry/output/basin_smb_anom_dmdt_1992_2017.npz', allow_pickle=True)
basin_smb_ref = basins_smb['basin_smb_ref']
basin_smb_anom_dmdt_1992_2017 = basins_smb['basin_smb_anom_dmdt_1992_2017']
basin_names = basins_smb['basin_names']

# partition as dynamics = dm - smb
discharge_ra_cpom = dmdt_ra_cpom - basin_smb_anom_dmdt_1992_2017

# convert partitioned discharge from anomaly to absolute values
discharge_ra_cpom_abs = basin_smb_ref - discharge_ra_cpom

# adjust for differences in reference smb
# need Rignot reference SMB for each basin
# smb_rignot = pd.read_excel('/Users/thomas/Documents/github/imbie_partitioning/aux/iom_datasets/pnas.1812883116.sd01_basins_discharge_1992_2017.xlsx',sheet_name='basins_smb_ref_1992_2017')
# smb_ref_rignot = smb_rignot['SMB_ref (Gt/yr)'].values
# discharge_ra_cpom_abs = discharge_ra_cpom_abs - (basin_smb_ref - smb_ref_rignot)


# =============================================================================
# 4. Compare to IOM
# =============================================================================

fig = plt.figure(figsize=(7, 3.5), constrained_layout=True)
gs = plt.GridSpec(1, 2, figure=fig, wspace=0.1)

# scatter
ax1 = fig.add_subplot(gs[0])

# 1 to 1 line
plt.plot([0, 300], [0, 300],
         linewidth=0.5,
         color='k')

ax1.scatter(discharge_rignot_basins, discharge_ra_cpom_abs, alpha=0.5)

for i, txt in enumerate(np.arange(0, len(discharge_rignot_basins)) + 1):
    ax1.annotate(
        str(txt), (discharge_rignot_basins[i], discharge_ra_cpom_abs[i]))

ax1.set_xlim((0, 300))
ax1.set_ylim((0, 300))

ax1.set_aspect('equal', 'box')

plt.xlabel('Discharge IOM [Gt/yr]')
plt.ylabel('Partitioned discharge RA [Gt/yr]')

# statistics
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(
    discharge_rignot_basins.astype(float), discharge_ra_cpom_abs.astype(float))
plt.text(290, 50,
         "R$^2$ = %.2f" % r_value ** 2,
         ha='right')

plt.text(290, 25,
         "RMSE = %.2f Gt/yr" % mean_squared_error(
             discharge_rignot_basins.astype(float), discharge_ra_cpom_abs.astype(float), squared=False),
         ha='right')

# map
ax2 = fig.add_subplot(gs[1])

ax2.fill(cx, cy, color='lightgray', linewidth=0.5)

for geom in gl_coords.geoms:
    gx, gy = geom.exterior.xy
    ax2.fill(gx, gy, color='darkgray', linewidth=0.5)

ax2.set_aspect('equal')

cmap = plt.cm.RdBu
# normalised by basin area
norm = plt.Normalize(-50, 50)
patches = []

for i, basin in enumerate(basins):
    region_idx = basins_gdf[basins_gdf['NAME'] ==
                            basin].index  # get basins for desired region
    if i == 12:
        basin_coords = list(
            basins_gdf['geometry'][region_idx[0]][1].exterior.coords)
    else:
        basin_coords = list(
            basins_gdf['geometry'][region_idx[0]].exterior.coords)
    basin_coords = np.array(basin_coords)
    bx = basin_coords[:, 0]
    by = basin_coords[:, 1]

    # plot as Gt/yr
    color = cmap(norm((discharge_rignot_basins[i] - discharge_ra_cpom_abs[i])))
    patches.append(
        Polygon(np.array(np.transpose([bx, by])), True, color=color))


pc = PatchCollection(patches, match_original=True,
                     edgecolor='none', linewidths=1., zorder=2, rasterized=True)
ax2.add_collection(pc)

for i, basin in enumerate(basins):
    region_idx = basins_gdf[basins_gdf['NAME'] ==
                            basin].index  # get basins for desired region
    if i == 12:
        basin_coords = list(
            basins_gdf['geometry'][region_idx[0]][1].exterior.coords)
    else:
        basin_coords = list(
            basins_gdf['geometry'][region_idx[0]].exterior.coords)
    basin_coords = np.array(basin_coords)
    bx = basin_coords[:, 0]
    by = basin_coords[:, 1]
    ax2.plot(bx, by, color='k', linewidth=0.05)

    # add number label
    if i == 9 or i == 10 or i == 11:
        ax2.annotate(str(i+1), (basins_gdf['geometry'][region_idx[0]].representative_point(
        ).xy[0][0]-0.25e6, basins_gdf['geometry'][region_idx[0]].representative_point().xy[1][0]))
    elif i == 12 or i == 17:
        ax2.annotate(str(i+1), (basins_gdf['geometry'][region_idx[0]].representative_point(
        ).xy[0][0]-0.5e6, basins_gdf['geometry'][region_idx[0]].representative_point().xy[1][0]))
    else:
        ax2.annotate(str(i+1), (basins_gdf['geometry'][region_idx[0]].representative_point(
        ).xy[0][0], basins_gdf['geometry'][region_idx[0]].representative_point().xy[1][0]))

ax2.axis('off')

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array(discharge_rignot_basins.astype(
    float) - discharge_ra_cpom_abs.astype(float))

# colorbar
cax = ax2.inset_axes([0.1, 0.15, 0.35, 0.05])
clb = plt.colorbar(sm,
                   cax=cax,
                   orientation='horizontal',
                   extend='both')
clb.ax.set_title('[Gt/yr]', position=(-0.2, -5))

# save
plt.savefig('/Users/thomas/Documents/github/imbie_partitioning/figs/discharge_compare_RA_IOM_ais.svg',
            format='svg', dpi=600, bbox_inches='tight')

# =============================================================================
# 5. Tabulate discharge rates per technique per ice sheet
# =============================================================================
# =============================================================================
# Antarctica
# =============================================================================
# load imbie partitioned data
imbie_ais = pd.read_csv('/Users/thomas/Documents/github/imbie_partitioning/partitioned_data/imbie_antarctica_2021_Gt_partitioned.csv',
                        float_precision='round_trip')
time_imbie_ais = imbie_ais['Year']
dmdt_smb_imbie_ais = imbie_ais['Surface mass balance anomaly (Gt/yr)']
dmdt_smb_uncert_imbie_ais = imbie_ais[
    'Surface mass balance anomaly uncertainty (Gt/yr)']

# load aggregated time series for indivdual techniques
techniques_ais = pd.read_csv('/Users/thomas/Documents/github/imbie_partitioning/aux/imbie_datasets/individual_techniques_aggregated/ais.csv',
                             float_precision='round_trip',
                             header=None)

# Altimetry
time_ra_ais = techniques_ais[2].iloc[np.where(
    techniques_ais[0] == 'RA')].values
dmdt_ra_ais = techniques_ais[3].iloc[np.where(
    techniques_ais[0] == 'RA')].values
dmdt_uncert_ra_ais = techniques_ais[4].iloc[np.where(
    techniques_ais[0] == 'RA')].values

# GMB
time_gmb_ais = techniques_ais[2].iloc[np.where(
    techniques_ais[0] == 'GMB')].values
dmdt_gmb_ais = techniques_ais[3].iloc[np.where(
    techniques_ais[0] == 'GMB')].values
dmdt_uncert_gmb_ais = techniques_ais[4].iloc[np.where(
    techniques_ais[0] == 'GMB')].values

# compute discharge for Altimetry and GMB
# Altimetry
# interpolate SMB
dmdt_smb_ra_ais = np.interp(time_ra_ais.astype(float),
                            time_imbie_ais.astype(float),
                            dmdt_smb_imbie_ais.astype(float))

dmdt_smb_uncert_ra_ais = np.interp(time_ra_ais.astype(float),
                                   time_imbie_ais.astype(float),
                                   dmdt_smb_uncert_imbie_ais.astype(float))

dmdt_dyn_ra_ais = dmdt_ra_ais.astype(float) - dmdt_smb_ra_ais.astype(float)
dmdt_dyn_uncert_ra_ais = np.sqrt(dmdt_uncert_ra_ais.astype(
    float) ** 2 + (dmdt_smb_uncert_ra_ais.astype(float) / 12) ** 2)

# compute annual discharge
time_ra_ais_annual = []
dmdt_dyn_ra_ais_annual = []
dmdt_dyn_uncert_ra_ais_annual = []

for i, year in enumerate(np.unique(np.floor(time_ra_ais.astype(float)))):
    dmdts_in_year = dmdt_dyn_ra_ais[np.where(
        np.floor(time_ra_ais.astype(float)) == year)]
    dmdt_uncerts_in_year = dmdt_dyn_uncert_ra_ais[np.where(
        np.floor(time_ra_ais.astype(float)) == year)]
    if len(dmdts_in_year) == 12:
        time_ra_ais_annual.append(year)
        dmdt_dyn_ra_ais_annual.append(dmdts_in_year.sum() / 12)
        dmdt_dyn_uncert_ra_ais_annual.append(
            np.sqrt((dmdt_uncerts_in_year ** 2).sum() / np.sqrt(12)))
    del dmdts_in_year

time_ra_ais_annual = np.array(time_ra_ais_annual, dtype=float)
dmdt_dyn_ra_ais_annual = np.array(dmdt_dyn_ra_ais_annual, dtype=float)
dmdt_dyn_uncert_ra_ais_annual = np.array(
    dmdt_dyn_uncert_ra_ais_annual, dtype=float)

# GMB
# interpolate SMB
dmdt_smb_gmb_ais = np.interp(time_gmb_ais.astype(float),
                             time_imbie_ais.astype(float),
                             dmdt_smb_imbie_ais.astype(float))

dmdt_smb_uncert_gmb_ais = np.interp(time_gmb_ais.astype(float),
                                    time_imbie_ais.astype(float),
                                    dmdt_smb_uncert_imbie_ais.astype(float))

dmdt_dyn_gmb_ais = dmdt_gmb_ais.astype(float) - dmdt_smb_gmb_ais.astype(float)
dmdt_dyn_uncert_gmb_ais = np.sqrt(dmdt_uncert_gmb_ais.astype(
    float) ** 2 + (dmdt_smb_uncert_gmb_ais.astype(float) / 12) ** 2)

# compute annual discharge
time_gmb_ais_annual = []
dmdt_dyn_gmb_ais_annual = []
dmdt_dyn_uncert_gmb_ais_annual = []

for i, year in enumerate(np.unique(np.floor(time_gmb_ais.astype(float)))):
    dmdts_in_year = dmdt_dyn_gmb_ais[np.where(
        np.floor(time_gmb_ais.astype(float)) == year)]
    dmdt_uncerts_in_year = dmdt_dyn_uncert_gmb_ais[np.where(
        np.floor(time_gmb_ais.astype(float)) == year)]
    if len(dmdts_in_year) == 12:
        time_gmb_ais_annual.append(year)
        dmdt_dyn_gmb_ais_annual.append(dmdts_in_year.sum() / 12)
        dmdt_dyn_uncert_gmb_ais_annual.append(
            np.sqrt((dmdt_uncerts_in_year ** 2).sum() / np.sqrt(12)))
    del dmdts_in_year

time_gmb_ais_annual = np.array(time_gmb_ais_annual, dtype=float)
dmdt_dyn_gmb_ais_annual = np.array(dmdt_dyn_gmb_ais_annual, dtype=float)
dmdt_dyn_uncert_gmb_ais_annual = np.array(
    dmdt_dyn_uncert_gmb_ais_annual, dtype=float)

# load smb reference period to convert partitioned discharge from anomaly to absolute values
smb_ref_ais = np.load(
    '/Users/thomas/Documents/github/imbie_partitioning/aux/smb_ref/smb_ref_ais.npy')
dmdt_dyn_ra_ais_annual_abs = smb_ref_ais - dmdt_dyn_ra_ais_annual
dmdt_dyn_uncert_ra_ais_annual_abs = np.sqrt(
    (smb_ref_ais * 0.2) ** 2 + dmdt_dyn_uncert_ra_ais ** 2)
dmdt_dyn_gmb_ais_annual_abs = smb_ref_ais - dmdt_dyn_gmb_ais_annual
dmdt_dyn_uncert_gmb_ais_annual_abs = np.sqrt(
    (smb_ref_ais * 0.2) ** 2 + dmdt_dyn_uncert_gmb_ais ** 2)

# load IOM discharge
iom = pd.read_csv('/Users/thomas/Documents/github/imbie_partitioning/aux/iom_datasets/pnas.1812883116.sd01_discharge.csv',
                  float_precision='round_trip')
time_iom = iom['Year'].values
dmdt_dyn_iom_ais = iom['AIS discharge'].values
dmdt_dyn_uncert_iom_ais = iom['AIS discharge uncertainty'].values

# find overlap
time_ra_gmb_ais, idx_ra_ais, idx_gmb_ais = np.intersect1d(
    time_ra_ais_annual, time_gmb_ais_annual, return_indices=True)

time_iom_overlap_ais, idx_iom_ais, unused = np.intersect1d(
    time_iom, time_ra_gmb_ais, return_indices=True)

# get mean for overlap period (2002 - 2017)
discharge_ais_ra_overlap = dmdt_dyn_ra_ais_annual_abs[idx_ra_ais].mean()
discharge_uncert_ais_ra_overlap = np.sqrt(
    (dmdt_dyn_uncert_ra_ais_annual_abs[idx_ra_ais] ** 2).sum()) / np.sqrt(len(idx_ra_ais))

discharge_ais_gmb_overlap = dmdt_dyn_gmb_ais_annual_abs[idx_gmb_ais].mean()
discharge_uncert_ais_gmb_overlap = np.sqrt(
    (dmdt_dyn_uncert_gmb_ais_annual_abs[idx_gmb_ais] ** 2).sum()) / np.sqrt(len(idx_gmb_ais))

discharge_ais_iom_overlap = dmdt_dyn_iom_ais[idx_iom_ais].mean()
discharge_uncert_ais_iom_overlap = np.sqrt(
    (dmdt_dyn_uncert_iom_ais[idx_iom_ais] ** 2).sum()) / np.sqrt(len(idx_iom_ais))

# =============================================================================
# West Antarctica
# =============================================================================
# load imbie partitioned data
imbie_wais = pd.read_csv('/Users/thomas/Documents/github/imbie_partitioning/partitioned_data/imbie_antarctica_2021_Gt_partitioned.csv',
                         float_precision='round_trip')
time_imbie_wais = imbie_wais['Year']
dmdt_smb_imbie_wais = imbie_wais['Surface mass balance anomaly (Gt/yr)']
dmdt_smb_uncert_imbie_wais = imbie_wais[
    'Surface mass balance anomaly uncertainty (Gt/yr)']

# load aggregated time series for indivdual techniques
techniques_wais = pd.read_csv('/Users/thomas/Documents/github/imbie_partitioning/aux/imbie_datasets/individual_techniques_aggregated/wais.csv',
                              float_precision='round_trip',
                              header=None)

# Altimetry
time_ra_wais = techniques_wais[2].iloc[np.where(
    techniques_wais[0] == 'RA')].values
dmdt_ra_wais = techniques_wais[3].iloc[np.where(
    techniques_wais[0] == 'RA')].values
dmdt_uncert_ra_wais = techniques_wais[4].iloc[np.where(
    techniques_wais[0] == 'RA')].values

# GMB
time_gmb_wais = techniques_wais[2].iloc[np.where(
    techniques_wais[0] == 'GMB')].values
dmdt_gmb_wais = techniques_wais[3].iloc[np.where(
    techniques_wais[0] == 'GMB')].values
dmdt_uncert_gmb_wais = techniques_wais[4].iloc[np.where(
    techniques_wais[0] == 'GMB')].values

# compute discharge for Altimetry and GMB
# Altimetry
# interpolate SMB
dmdt_smb_ra_wais = np.interp(time_ra_wais.astype(float),
                             time_imbie_wais.astype(float),
                             dmdt_smb_imbie_wais.astype(float))

dmdt_smb_uncert_ra_wais = np.interp(time_ra_wais.astype(float),
                                    time_imbie_wais.astype(float),
                                    dmdt_smb_uncert_imbie_wais.astype(float))

dmdt_dyn_ra_wais = dmdt_ra_wais.astype(float) - dmdt_smb_ra_wais.astype(float)
dmdt_dyn_uncert_ra_wais = np.sqrt(dmdt_uncert_ra_wais.astype(
    float) ** 2 + (dmdt_smb_uncert_ra_wais.astype(float) / 12) ** 2)

# compute annual discharge
time_ra_wais_annual = []
dmdt_dyn_ra_wais_annual = []
dmdt_dyn_uncert_ra_wais_annual = []

for i, year in enumerate(np.unique(np.floor(time_ra_wais.astype(float)))):
    dmdts_in_year = dmdt_dyn_ra_wais[np.where(
        np.floor(time_ra_wais.astype(float)) == year)]
    dmdt_uncerts_in_year = dmdt_dyn_uncert_ra_wais[np.where(
        np.floor(time_ra_wais.astype(float)) == year)]
    if len(dmdts_in_year) == 12:
        time_ra_wais_annual.append(year)
        dmdt_dyn_ra_wais_annual.append(dmdts_in_year.sum() / 12)
        dmdt_dyn_uncert_ra_wais_annual.append(
            np.sqrt((dmdt_uncerts_in_year ** 2).sum() / np.sqrt(12)))
    del dmdts_in_year

time_ra_wais_annual = np.array(time_ra_wais_annual, dtype=float)
dmdt_dyn_ra_wais_annual = np.array(dmdt_dyn_ra_wais_annual, dtype=float)
dmdt_dyn_uncert_ra_wais_annual = np.array(
    dmdt_dyn_uncert_ra_wais_annual, dtype=float)

# GMB
# interpolate SMB
dmdt_smb_gmb_wais = np.interp(time_gmb_wais.astype(float),
                              time_imbie_wais.astype(float),
                              dmdt_smb_imbie_wais.astype(float))

dmdt_smb_uncert_gmb_wais = np.interp(time_gmb_wais.astype(float),
                                     time_imbie_wais.astype(float),
                                     dmdt_smb_uncert_imbie_wais.astype(float))

dmdt_dyn_gmb_wais = dmdt_gmb_wais.astype(
    float) - dmdt_smb_gmb_wais.astype(float)
dmdt_dyn_uncert_gmb_wais = np.sqrt(dmdt_uncert_gmb_wais.astype(
    float) ** 2 + (dmdt_smb_uncert_gmb_wais.astype(float) / 12) ** 2)

# compute annual discharge
time_gmb_wais_annual = []
dmdt_dyn_gmb_wais_annual = []
dmdt_dyn_uncert_gmb_wais_annual = []

for i, year in enumerate(np.unique(np.floor(time_gmb_wais.astype(float)))):
    dmdts_in_year = dmdt_dyn_gmb_wais[np.where(
        np.floor(time_gmb_wais.astype(float)) == year)]
    dmdt_uncerts_in_year = dmdt_dyn_uncert_gmb_wais[np.where(
        np.floor(time_gmb_wais.astype(float)) == year)]
    if len(dmdts_in_year) == 12:
        time_gmb_wais_annual.append(year)
        dmdt_dyn_gmb_wais_annual.append(dmdts_in_year.sum() / 12)
        dmdt_dyn_uncert_gmb_wais_annual.append(
            np.sqrt((dmdt_uncerts_in_year ** 2).sum() / np.sqrt(12)))
    del dmdts_in_year

time_gmb_wais_annual = np.array(time_gmb_wais_annual, dtype=float)
dmdt_dyn_gmb_wais_annual = np.array(dmdt_dyn_gmb_wais_annual, dtype=float)
dmdt_dyn_uncert_gmb_wais_annual = np.array(
    dmdt_dyn_uncert_gmb_wais_annual, dtype=float)

# load smb reference period to convert partitioned discharge from anomaly to absolute values
smb_ref_wais = np.load(
    '/Users/thomas/Documents/github/imbie_partitioning/aux/smb_ref/smb_ref_wais.npy')
dmdt_dyn_ra_wais_annual_abs = smb_ref_wais - dmdt_dyn_ra_wais_annual
dmdt_dyn_uncert_ra_wais_annual_abs = np.sqrt(
    (smb_ref_wais * 0.2) ** 2 + dmdt_dyn_uncert_ra_wais ** 2)
dmdt_dyn_gmb_wais_annual_abs = smb_ref_wais - dmdt_dyn_gmb_wais_annual
dmdt_dyn_uncert_gmb_wais_annual_abs = np.sqrt(
    (smb_ref_wais * 0.2) ** 2 + dmdt_dyn_uncert_gmb_wais ** 2)

# load IOM discharge
iom = pd.read_csv('/Users/thomas/Documents/github/imbie_partitioning/aux/iom_datasets/pnas.1812883116.sd01_discharge.csv',
                  float_precision='round_trip')
time_iom = iom['Year'].values
dmdt_dyn_iom_wais = iom['WAIS discharge'].values
dmdt_dyn_uncert_iom_wais = iom['WAIS discharge uncertainty'].values

# find overlap
time_ra_gmb_wais, idx_ra_wais, idx_gmb_wais = np.intersect1d(
    time_ra_wais_annual, time_gmb_wais_annual, return_indices=True)

time_iom_overlap_wais, idx_iom_wais, unused = np.intersect1d(
    time_iom, time_ra_gmb_wais, return_indices=True)

# get mean for overlap period (2002 - 2017)
discharge_wais_ra_overlap = dmdt_dyn_ra_wais_annual_abs[idx_ra_wais].mean()
discharge_uncert_wais_ra_overlap = np.sqrt(
    (dmdt_dyn_uncert_ra_wais_annual_abs[idx_ra_wais] ** 2).sum()) / np.sqrt(len(idx_ra_wais))

discharge_wais_gmb_overlap = dmdt_dyn_gmb_wais_annual_abs[idx_gmb_wais].mean()
discharge_uncert_wais_gmb_overlap = np.sqrt(
    (dmdt_dyn_uncert_gmb_wais_annual_abs[idx_gmb_wais] ** 2).sum()) / np.sqrt(len(idx_gmb_wais))

discharge_wais_iom_overlap = dmdt_dyn_iom_wais[idx_iom_wais].mean()
discharge_uncert_wais_iom_overlap = np.sqrt(
    (dmdt_dyn_uncert_iom_wais[idx_iom_wais] ** 2).sum()) / np.sqrt(len(idx_iom_wais))

# =============================================================================
# East Antarctica
# =============================================================================
# load imbie partitioned data
imbie_eais = pd.read_csv('/Users/thomas/Documents/github/imbie_partitioning/partitioned_data/imbie_antarctica_2021_Gt_partitioned.csv',
                         float_precision='round_trip')
time_imbie_eais = imbie_eais['Year']
dmdt_smb_imbie_eais = imbie_eais['Surface mass balance anomaly (Gt/yr)']
dmdt_smb_uncert_imbie_eais = imbie_eais[
    'Surface mass balance anomaly uncertainty (Gt/yr)']

# load aggregated time series for indivdual techniques
techniques_eais = pd.read_csv('/Users/thomas/Documents/github/imbie_partitioning/aux/imbie_datasets/individual_techniques_aggregated/eais.csv',
                              float_precision='round_trip',
                              header=None)

# Altimetry
time_ra_eais = techniques_eais[2].iloc[np.where(
    techniques_eais[0] == 'RA')].values
dmdt_ra_eais = techniques_eais[3].iloc[np.where(
    techniques_eais[0] == 'RA')].values
dmdt_uncert_ra_eais = techniques_eais[4].iloc[np.where(
    techniques_eais[0] == 'RA')].values

# GMB
time_gmb_eais = techniques_eais[2].iloc[np.where(
    techniques_eais[0] == 'GMB')].values
dmdt_gmb_eais = techniques_eais[3].iloc[np.where(
    techniques_eais[0] == 'GMB')].values
dmdt_uncert_gmb_eais = techniques_eais[4].iloc[np.where(
    techniques_eais[0] == 'GMB')].values

# compute discharge for Altimetry and GMB
# Altimetry
# interpolate SMB
dmdt_smb_ra_eais = np.interp(time_ra_eais.astype(float),
                             time_imbie_eais.astype(float),
                             dmdt_smb_imbie_eais.astype(float))

dmdt_smb_uncert_ra_eais = np.interp(time_ra_eais.astype(float),
                                    time_imbie_eais.astype(float),
                                    dmdt_smb_uncert_imbie_eais.astype(float))

dmdt_dyn_ra_eais = dmdt_ra_eais.astype(float) - dmdt_smb_ra_eais.astype(float)
dmdt_dyn_uncert_ra_eais = np.sqrt(dmdt_uncert_ra_eais.astype(
    float) ** 2 + (dmdt_smb_uncert_ra_eais.astype(float) / 12) ** 2)

# compute annual discharge
time_ra_eais_annual = []
dmdt_dyn_ra_eais_annual = []
dmdt_dyn_uncert_ra_eais_annual = []

for i, year in enumerate(np.unique(np.floor(time_ra_eais.astype(float)))):
    dmdts_in_year = dmdt_dyn_ra_eais[np.where(
        np.floor(time_ra_eais.astype(float)) == year)]
    dmdt_uncerts_in_year = dmdt_dyn_uncert_ra_eais[np.where(
        np.floor(time_ra_eais.astype(float)) == year)]
    if len(dmdts_in_year) == 12:
        time_ra_eais_annual.append(year)
        dmdt_dyn_ra_eais_annual.append(dmdts_in_year.sum() / 12)
        dmdt_dyn_uncert_ra_eais_annual.append(
            np.sqrt((dmdt_uncerts_in_year ** 2).sum() / np.sqrt(12)))
    del dmdts_in_year

time_ra_eais_annual = np.array(time_ra_eais_annual, dtype=float)
dmdt_dyn_ra_eais_annual = np.array(dmdt_dyn_ra_eais_annual, dtype=float)
dmdt_dyn_uncert_ra_eais_annual = np.array(
    dmdt_dyn_uncert_ra_eais_annual, dtype=float)

# GMB
# interpolate SMB
dmdt_smb_gmb_eais = np.interp(time_gmb_eais.astype(float),
                              time_imbie_eais.astype(float),
                              dmdt_smb_imbie_eais.astype(float))

dmdt_smb_uncert_gmb_eais = np.interp(time_gmb_eais.astype(float),
                                     time_imbie_eais.astype(float),
                                     dmdt_smb_uncert_imbie_eais.astype(float))

dmdt_dyn_gmb_eais = dmdt_gmb_eais.astype(
    float) - dmdt_smb_gmb_eais.astype(float)
dmdt_dyn_uncert_gmb_eais = np.sqrt(dmdt_uncert_gmb_eais.astype(
    float) ** 2 + (dmdt_smb_uncert_gmb_eais.astype(float) / 12) ** 2)

# compute annual discharge
time_gmb_eais_annual = []
dmdt_dyn_gmb_eais_annual = []
dmdt_dyn_uncert_gmb_eais_annual = []

for i, year in enumerate(np.unique(np.floor(time_gmb_eais.astype(float)))):
    dmdts_in_year = dmdt_dyn_gmb_eais[np.where(
        np.floor(time_gmb_eais.astype(float)) == year)]
    dmdt_uncerts_in_year = dmdt_dyn_uncert_gmb_eais[np.where(
        np.floor(time_gmb_eais.astype(float)) == year)]
    if len(dmdts_in_year) == 12:
        time_gmb_eais_annual.append(year)
        dmdt_dyn_gmb_eais_annual.append(dmdts_in_year.sum() / 12)
        dmdt_dyn_uncert_gmb_eais_annual.append(
            np.sqrt((dmdt_uncerts_in_year ** 2).sum() / np.sqrt(12)))
    del dmdts_in_year

time_gmb_eais_annual = np.array(time_gmb_eais_annual, dtype=float)
dmdt_dyn_gmb_eais_annual = np.array(dmdt_dyn_gmb_eais_annual, dtype=float)
dmdt_dyn_uncert_gmb_eais_annual = np.array(
    dmdt_dyn_uncert_gmb_eais_annual, dtype=float)

# load smb reference period to convert partitioned discharge from anomaly to absolute values
smb_ref_eais = np.load(
    '/Users/thomas/Documents/github/imbie_partitioning/aux/smb_ref/smb_ref_eais.npy')
dmdt_dyn_ra_eais_annual_abs = smb_ref_eais - dmdt_dyn_ra_eais_annual
dmdt_dyn_uncert_ra_eais_annual_abs = np.sqrt(
    (smb_ref_eais * 0.2) ** 2 + dmdt_dyn_uncert_ra_eais ** 2)
dmdt_dyn_gmb_eais_annual_abs = smb_ref_eais - dmdt_dyn_gmb_eais_annual
dmdt_dyn_uncert_gmb_eais_annual_abs = np.sqrt(
    (smb_ref_eais * 0.2) ** 2 + dmdt_dyn_uncert_gmb_eais ** 2)

# load IOM discharge
iom = pd.read_csv('/Users/thomas/Documents/github/imbie_partitioning/aux/iom_datasets/pnas.1812883116.sd01_discharge.csv',
                  float_precision='round_trip')
time_iom = iom['Year'].values
dmdt_dyn_iom_eais = iom['EAIS discharge'].values
dmdt_dyn_uncert_iom_eais = iom['EAIS discharge uncertainty'].values

# find overlap
time_ra_gmb_eais, idx_ra_eais, idx_gmb_eais = np.intersect1d(
    time_ra_eais_annual, time_gmb_eais_annual, return_indices=True)

time_iom_overlap_eais, idx_iom_eais, unused = np.intersect1d(
    time_iom, time_ra_gmb_eais, return_indices=True)

# get mean for overlap period (2002 - 2017)
discharge_eais_ra_overlap = dmdt_dyn_ra_eais_annual_abs[idx_ra_eais].mean()
discharge_uncert_eais_ra_overlap = np.sqrt(
    (dmdt_dyn_uncert_ra_eais_annual_abs[idx_ra_eais] ** 2).sum()) / np.sqrt(len(idx_ra_eais))

discharge_eais_gmb_overlap = dmdt_dyn_gmb_eais_annual_abs[idx_gmb_eais].mean()
discharge_uncert_eais_gmb_overlap = np.sqrt(
    (dmdt_dyn_uncert_gmb_eais_annual_abs[idx_gmb_eais] ** 2).sum()) / np.sqrt(len(idx_gmb_eais))

discharge_eais_iom_overlap = dmdt_dyn_iom_eais[idx_iom_eais].mean()
discharge_uncert_eais_iom_overlap = np.sqrt(
    (dmdt_dyn_uncert_iom_eais[idx_iom_eais] ** 2).sum()) / np.sqrt(len(idx_iom_eais))

# =============================================================================
# Antarctic Peninsula
# =============================================================================
# load imbie partitioned data
imbie_apis = pd.read_csv('/Users/thomas/Documents/github/imbie_partitioning/partitioned_data/imbie_antarctica_2021_Gt_partitioned.csv',
                         float_precision='round_trip')
time_imbie_apis = imbie_apis['Year']
dmdt_smb_imbie_apis = imbie_apis['Surface mass balance anomaly (Gt/yr)']
dmdt_smb_uncert_imbie_apis = imbie_apis[
    'Surface mass balance anomaly uncertainty (Gt/yr)']

# load aggregated time series for indivdual techniques
techniques_apis = pd.read_csv('/Users/thomas/Documents/github/imbie_partitioning/aux/imbie_datasets/individual_techniques_aggregated/apis.csv',
                              float_precision='round_trip',
                              header=None)

# Altimetry
time_ra_apis = techniques_apis[2].iloc[np.where(
    techniques_apis[0] == 'RA')].values
dmdt_ra_apis = techniques_apis[3].iloc[np.where(
    techniques_apis[0] == 'RA')].values
dmdt_uncert_ra_apis = techniques_apis[4].iloc[np.where(
    techniques_apis[0] == 'RA')].values

# GMB
time_gmb_apis = techniques_apis[2].iloc[np.where(
    techniques_apis[0] == 'GMB')].values
dmdt_gmb_apis = techniques_apis[3].iloc[np.where(
    techniques_apis[0] == 'GMB')].values
dmdt_uncert_gmb_apis = techniques_apis[4].iloc[np.where(
    techniques_apis[0] == 'GMB')].values

# compute discharge for Altimetry and GMB
# Altimetry
# interpolate SMB
dmdt_smb_ra_apis = np.interp(time_ra_apis.astype(float),
                             time_imbie_apis.astype(float),
                             dmdt_smb_imbie_apis.astype(float))

dmdt_smb_uncert_ra_apis = np.interp(time_ra_apis.astype(float),
                                    time_imbie_apis.astype(float),
                                    dmdt_smb_uncert_imbie_apis.astype(float))

dmdt_dyn_ra_apis = dmdt_ra_apis.astype(float) - dmdt_smb_ra_apis.astype(float)
dmdt_dyn_uncert_ra_apis = np.sqrt(dmdt_uncert_ra_apis.astype(
    float) ** 2 + (dmdt_smb_uncert_ra_apis.astype(float) / 12) ** 2)

# compute annual discharge
time_ra_apis_annual = []
dmdt_dyn_ra_apis_annual = []
dmdt_dyn_uncert_ra_apis_annual = []

for i, year in enumerate(np.unique(np.floor(time_ra_apis.astype(float)))):
    dmdts_in_year = dmdt_dyn_ra_apis[np.where(
        np.floor(time_ra_apis.astype(float)) == year)]
    dmdt_uncerts_in_year = dmdt_dyn_uncert_ra_apis[np.where(
        np.floor(time_ra_apis.astype(float)) == year)]
    if len(dmdts_in_year) == 12:
        time_ra_apis_annual.append(year)
        dmdt_dyn_ra_apis_annual.append(dmdts_in_year.sum() / 12)
        dmdt_dyn_uncert_ra_apis_annual.append(
            np.sqrt((dmdt_uncerts_in_year ** 2).sum() / np.sqrt(12)))
    del dmdts_in_year

time_ra_apis_annual = np.array(time_ra_apis_annual, dtype=float)
dmdt_dyn_ra_apis_annual = np.array(dmdt_dyn_ra_apis_annual, dtype=float)
dmdt_dyn_uncert_ra_apis_annual = np.array(
    dmdt_dyn_uncert_ra_apis_annual, dtype=float)

# GMB
# interpolate SMB
dmdt_smb_gmb_apis = np.interp(time_gmb_apis.astype(float),
                              time_imbie_apis.astype(float),
                              dmdt_smb_imbie_apis.astype(float))

dmdt_smb_uncert_gmb_apis = np.interp(time_gmb_apis.astype(float),
                                     time_imbie_apis.astype(float),
                                     dmdt_smb_uncert_imbie_apis.astype(float))

dmdt_dyn_gmb_apis = dmdt_gmb_apis.astype(
    float) - dmdt_smb_gmb_apis.astype(float)
dmdt_dyn_uncert_gmb_apis = np.sqrt(dmdt_uncert_gmb_apis.astype(
    float) ** 2 + (dmdt_smb_uncert_gmb_apis.astype(float) / 12) ** 2)

# compute annual discharge
time_gmb_apis_annual = []
dmdt_dyn_gmb_apis_annual = []
dmdt_dyn_uncert_gmb_apis_annual = []

for i, year in enumerate(np.unique(np.floor(time_gmb_apis.astype(float)))):
    dmdts_in_year = dmdt_dyn_gmb_apis[np.where(
        np.floor(time_gmb_apis.astype(float)) == year)]
    dmdt_uncerts_in_year = dmdt_dyn_uncert_gmb_apis[np.where(
        np.floor(time_gmb_apis.astype(float)) == year)]
    if len(dmdts_in_year) == 12:
        time_gmb_apis_annual.append(year)
        dmdt_dyn_gmb_apis_annual.append(dmdts_in_year.sum() / 12)
        dmdt_dyn_uncert_gmb_apis_annual.append(
            np.sqrt((dmdt_uncerts_in_year ** 2).sum() / np.sqrt(12)))
    del dmdts_in_year

time_gmb_apis_annual = np.array(time_gmb_apis_annual, dtype=float)
dmdt_dyn_gmb_apis_annual = np.array(dmdt_dyn_gmb_apis_annual, dtype=float)
dmdt_dyn_uncert_gmb_apis_annual = np.array(
    dmdt_dyn_uncert_gmb_apis_annual, dtype=float)

# load smb reference period to convert partitioned discharge from anomaly to absolute values
smb_ref_apis = np.load(
    '/Users/thomas/Documents/github/imbie_partitioning/aux/smb_ref/smb_ref_apis.npy')
dmdt_dyn_ra_apis_annual_abs = smb_ref_apis - dmdt_dyn_ra_apis_annual
dmdt_dyn_uncert_ra_apis_annual_abs = np.sqrt(
    (smb_ref_apis * 0.2) ** 2 + dmdt_dyn_uncert_ra_apis ** 2)
dmdt_dyn_gmb_apis_annual_abs = smb_ref_apis - dmdt_dyn_gmb_apis_annual
dmdt_dyn_uncert_gmb_apis_annual_abs = np.sqrt(
    (smb_ref_apis * 0.2) ** 2 + dmdt_dyn_uncert_gmb_apis ** 2)

# load IOM discharge
iom = pd.read_csv('/Users/thomas/Documents/github/imbie_partitioning/aux/iom_datasets/pnas.1812883116.sd01_discharge.csv',
                  float_precision='round_trip')
time_iom = iom['Year'].values
dmdt_dyn_iom_apis = iom['APIS discharge'].values
dmdt_dyn_uncert_iom_apis = iom['APIS discharge uncertainty'].values

# find overlap
time_ra_gmb_apis, idx_ra_apis, idx_gmb_apis = np.intersect1d(
    time_ra_apis_annual, time_gmb_apis_annual, return_indices=True)

time_iom_overlap_apis, idx_iom_apis, unused = np.intersect1d(
    time_iom, time_ra_gmb_apis, return_indices=True)

# get mean for overlap period (2002 - 2017)
discharge_apis_ra_overlap = dmdt_dyn_ra_apis_annual_abs[idx_ra_apis].mean()
discharge_uncert_apis_ra_overlap = np.sqrt(
    (dmdt_dyn_uncert_ra_apis_annual_abs[idx_ra_apis] ** 2).sum()) / np.sqrt(len(idx_ra_apis))

discharge_apis_gmb_overlap = dmdt_dyn_gmb_apis_annual_abs[idx_gmb_apis].mean()
discharge_uncert_apis_gmb_overlap = np.sqrt(
    (dmdt_dyn_uncert_gmb_apis_annual_abs[idx_gmb_apis] ** 2).sum()) / np.sqrt(len(idx_gmb_apis))

discharge_apis_iom_overlap = dmdt_dyn_iom_apis[idx_iom_apis].mean()
discharge_uncert_apis_iom_overlap = np.sqrt(
    (dmdt_dyn_uncert_iom_apis[idx_iom_apis] ** 2).sum()) / np.sqrt(len(idx_iom_apis))

# =============================================================================
# Greenland
# =============================================================================
# load imbie partitioned data
imbie_gris = pd.read_csv('/Users/thomas/Documents/github/imbie_partitioning/partitioned_data/imbie_antarctica_2021_Gt_partitioned.csv',
                         float_precision='round_trip')
time_imbie_gris = imbie_gris['Year']
dmdt_smb_imbie_gris = imbie_gris['Surface mass balance anomaly (Gt/yr)']
dmdt_smb_uncert_imbie_gris = imbie_gris[
    'Surface mass balance anomaly uncertainty (Gt/yr)']

# load aggregated time series for indivdual techniques
techniques_gris = pd.read_csv('/Users/thomas/Documents/github/imbie_partitioning/aux/imbie_datasets/individual_techniques_aggregated/gris.csv',
                              float_precision='round_trip',
                              header=None)

# Altimetry
time_ra_gris = techniques_gris[2].iloc[np.where(
    techniques_gris[0] == 'RA')].values
dmdt_ra_gris = techniques_gris[3].iloc[np.where(
    techniques_gris[0] == 'RA')].values
dmdt_uncert_ra_gris = techniques_gris[4].iloc[np.where(
    techniques_gris[0] == 'RA')].values

# GMB
time_gmb_gris = techniques_gris[2].iloc[np.where(
    techniques_gris[0] == 'GMB')].values
dmdt_gmb_gris = techniques_gris[3].iloc[np.where(
    techniques_gris[0] == 'GMB')].values
dmdt_uncert_gmb_gris = techniques_gris[4].iloc[np.where(
    techniques_gris[0] == 'GMB')].values

# compute discharge for Altimetry and GMB
# Altimetry
# interpolate SMB
dmdt_smb_ra_gris = np.interp(time_ra_gris.astype(float),
                             time_imbie_gris.astype(float),
                             dmdt_smb_imbie_gris.astype(float))

dmdt_smb_uncert_ra_gris = np.interp(time_ra_gris.astype(float),
                                    time_imbie_gris.astype(float),
                                    dmdt_smb_uncert_imbie_gris.astype(float))

dmdt_dyn_ra_gris = dmdt_ra_gris.astype(float) - dmdt_smb_ra_gris.astype(float)
dmdt_dyn_uncert_ra_gris = np.sqrt(dmdt_uncert_ra_gris.astype(
    float) ** 2 + (dmdt_smb_uncert_ra_gris.astype(float) / 12) ** 2)

# compute annual discharge
time_ra_gris_annual = []
dmdt_dyn_ra_gris_annual = []
dmdt_dyn_uncert_ra_gris_annual = []

for i, year in enumerate(np.unique(np.floor(time_ra_gris.astype(float)))):
    dmdts_in_year = dmdt_dyn_ra_gris[np.where(
        np.floor(time_ra_gris.astype(float)) == year)]
    dmdt_uncerts_in_year = dmdt_dyn_uncert_ra_gris[np.where(
        np.floor(time_ra_gris.astype(float)) == year)]
    if len(dmdts_in_year) == 12:
        time_ra_gris_annual.append(year)
        dmdt_dyn_ra_gris_annual.append(dmdts_in_year.sum() / 12)
        dmdt_dyn_uncert_ra_gris_annual.append(
            np.sqrt((dmdt_uncerts_in_year ** 2).sum() / np.sqrt(12)))
    del dmdts_in_year

time_ra_gris_annual = np.array(time_ra_gris_annual, dtype=float)
dmdt_dyn_ra_gris_annual = np.array(dmdt_dyn_ra_gris_annual, dtype=float)
dmdt_dyn_uncert_ra_gris_annual = np.array(
    dmdt_dyn_uncert_ra_gris_annual, dtype=float)

# GMB
# interpolate SMB
dmdt_smb_gmb_gris = np.interp(time_gmb_gris.astype(float),
                              time_imbie_gris.astype(float),
                              dmdt_smb_imbie_gris.astype(float))

dmdt_smb_uncert_gmb_gris = np.interp(time_gmb_gris.astype(float),
                                     time_imbie_gris.astype(float),
                                     dmdt_smb_uncert_imbie_gris.astype(float))

dmdt_dyn_gmb_gris = dmdt_gmb_gris.astype(
    float) - dmdt_smb_gmb_gris.astype(float)
dmdt_dyn_uncert_gmb_gris = np.sqrt(dmdt_uncert_gmb_gris.astype(
    float) ** 2 + (dmdt_smb_uncert_gmb_gris.astype(float) / 12) ** 2)

# compute annual discharge
time_gmb_gris_annual = []
dmdt_dyn_gmb_gris_annual = []
dmdt_dyn_uncert_gmb_gris_annual = []

for i, year in enumerate(np.unique(np.floor(time_gmb_gris.astype(float)))):
    dmdts_in_year = dmdt_dyn_gmb_gris[np.where(
        np.floor(time_gmb_gris.astype(float)) == year)]
    dmdt_uncerts_in_year = dmdt_dyn_uncert_gmb_gris[np.where(
        np.floor(time_gmb_gris.astype(float)) == year)]
    if len(dmdts_in_year) == 12:
        time_gmb_gris_annual.append(year)
        dmdt_dyn_gmb_gris_annual.append(dmdts_in_year.sum() / 12)
        dmdt_dyn_uncert_gmb_gris_annual.append(
            np.sqrt((dmdt_uncerts_in_year ** 2).sum() / np.sqrt(12)))
    del dmdts_in_year

time_gmb_gris_annual = np.array(time_gmb_gris_annual, dtype=float)
dmdt_dyn_gmb_gris_annual = np.array(dmdt_dyn_gmb_gris_annual, dtype=float)
dmdt_dyn_uncert_gmb_gris_annual = np.array(
    dmdt_dyn_uncert_gmb_gris_annual, dtype=float)

# load smb reference period to convert partitioned discharge from anomaly to absolute values
smb_ref_gris = np.load(
    '/Users/thomas/Documents/github/imbie_partitioning/aux/smb_ref/smb_ref_gris.npy')
dmdt_dyn_ra_gris_annual_abs = smb_ref_gris - dmdt_dyn_ra_gris_annual
dmdt_dyn_uncert_ra_gris_annual_abs = np.sqrt(
    (smb_ref_gris * 0.2) ** 2 + dmdt_dyn_uncert_ra_gris ** 2)
dmdt_dyn_gmb_gris_annual_abs = smb_ref_gris - dmdt_dyn_gmb_gris_annual
dmdt_dyn_uncert_gmb_gris_annual_abs = np.sqrt(
    (smb_ref_gris * 0.2) ** 2 + dmdt_dyn_uncert_gmb_gris ** 2)

# load IOM discharge
iom = pd.read_csv('/Users/thomas/Documents/github/imbie_partitioning/aux/iom_datasets/pnas.1904242116.sd02_discharge.csv',
                  float_precision='round_trip')
time_iom = iom['Year'].values
dmdt_dyn_iom_gris = iom['GRIS discharge'].values
dmdt_dyn_uncert_iom_gris = iom['GRIS discharge uncertainty'].values

# find overlap
time_ra_gmb_gris, idx_ra_gris, idx_gmb_gris = np.intersect1d(
    time_ra_gris_annual, time_gmb_gris_annual, return_indices=True)

time_iom_overlap_gris, idx_iom_gris, unused = np.intersect1d(
    time_iom, time_ra_gmb_gris, return_indices=True)

# get mean for overlap period (2004 - 2017)
discharge_gris_ra_overlap = dmdt_dyn_ra_gris_annual_abs[idx_ra_gris].mean()
discharge_uncert_gris_ra_overlap = np.sqrt(
    (dmdt_dyn_uncert_ra_gris_annual_abs[idx_ra_gris] ** 2).sum()) / np.sqrt(len(idx_ra_gris))

discharge_gris_gmb_overlap = dmdt_dyn_gmb_gris_annual_abs[idx_gmb_gris].mean()
discharge_uncert_gris_gmb_overlap = np.sqrt(
    (dmdt_dyn_uncert_gmb_gris_annual_abs[idx_gmb_gris] ** 2).sum()) / np.sqrt(len(idx_gmb_gris))

discharge_gris_iom_overlap = dmdt_dyn_iom_gris[idx_iom_gris].mean()
discharge_uncert_gris_iom_overlap = np.sqrt(
    (dmdt_dyn_uncert_iom_gris[idx_iom_gris] ** 2).sum()) / np.sqrt(len(idx_iom_gris))

# =============================================================================
# Output Table
# =============================================================================
m = np.array([['GRIS (2004 - 2017)', str(discharge_gris_ra_overlap.round(0))+' ± '+str(discharge_uncert_gris_ra_overlap.round(0)),
               str(discharge_gris_gmb_overlap.round(0))+' ± ' +
               str(discharge_uncert_gris_gmb_overlap.round(0)),
               str(discharge_gris_iom_overlap.round(0))+' ± '+str(discharge_uncert_gris_iom_overlap.round(0))],
              ['AIS (2002 - 2017)', str(discharge_ais_ra_overlap.round(0))+' ± '+str(discharge_uncert_ais_ra_overlap.round(0)),
               str(discharge_ais_gmb_overlap.round(0))+' ± ' +
             str(discharge_uncert_ais_gmb_overlap.round(0)),
               str(discharge_ais_iom_overlap.round(0))+' ± '+str(discharge_uncert_ais_iom_overlap.round(0))],
              ['WAIS (2002 - 2017)', str(discharge_wais_ra_overlap.round(0))+' ± '+str(discharge_uncert_wais_ra_overlap.round(0)),
               str(discharge_wais_gmb_overlap.round(0))+' ± ' +
               str(discharge_uncert_wais_gmb_overlap.round(0)),
               str(discharge_wais_iom_overlap.round(0))+' ± '+str(discharge_uncert_wais_iom_overlap.round(0))],
              ['EAIS (2002 - 2017)', str(discharge_eais_ra_overlap.round(0))+' ± '+str(discharge_uncert_eais_ra_overlap.round(0)),
               str(discharge_eais_gmb_overlap.round(0))+' ± ' +
               str(discharge_uncert_eais_gmb_overlap.round(0)),
               str(discharge_eais_iom_overlap.round(0))+' ± '+str(discharge_uncert_eais_iom_overlap.round(0))],
              ['APIS (2002 - 2017)', str(discharge_apis_ra_overlap.round(0))+' ± '+str(discharge_uncert_apis_ra_overlap.round(0)),
               str(discharge_apis_gmb_overlap.round(0))+' ± ' +
               str(discharge_uncert_apis_gmb_overlap.round(0)),
               str(discharge_apis_iom_overlap.round(0))+' ± '+str(discharge_uncert_apis_iom_overlap.round(0))]])
headers = ["RA (Gt/yr)", "GMB (Gt/yr)", "IOM (Gt/yr)"]

table = tabulate(m, headers, tablefmt="fancy_grid")
print(table)
