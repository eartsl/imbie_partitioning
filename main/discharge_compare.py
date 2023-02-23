#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 11:02:15 2023

@author: thomas
"""

# =============================================================================
# initialise
# =============================================================================

## imports ##

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sys
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from scipy import interpolate
import scipy.stats

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
discharge_rignot = pd.read_excel('/Users/thomas/Documents/github/imbie_partitioning/aux/iom_datasets/pnas.1812883116.sd01_basins_discharge_1992_2017.xlsx',sheet_name='basins_discharge_1992_2017')
discharge_rignot_basins = discharge_rignot['Average discharge (Gt/yr)'].values
# =============================================================================
# 2. find altimetry dM/dt in overlap period
# =============================================================================
# apply linear regression to each row to get dM/dt
dmdt_ra_cpom = np.full((np.shape(dm_ra_cpom)[0]), np.nan)
for i, row in enumerate(dm_ra_cpom):
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(
        time_ra_cpom[np.where(time_ra_cpom<2017)],
        row[np.where(time_ra_cpom<2017)])
    dmdt_ra_cpom[i] = slope

# =============================================================================
# 3. Partition into dynamics
# =============================================================================
# load smb anomaly dmdt during overlap period
basins_smb = np.load('/Users/thomas/Documents/github/imbie_partitioning/aux/altimetry/output/basin_smb_anom_dmdt_1992_2017.npz', allow_pickle=True)
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

fig = plt.figure(figsize=(4, 4), constrained_layout=True)
gs = plt.GridSpec(1, 1, figure=fig, wspace=0.1)

ax1 = fig.add_subplot(gs[0])

# 1 to 1 line
plt.plot([0, 350], [0, 350],
         linewidth=0.5,
         color='k')

ax1.scatter(discharge_rignot_basins,discharge_ra_cpom_abs)

ax1.set_xlim((0, 350))
ax1.set_ylim((0, 350))

ax1.set_aspect('equal', 'box')

plt.xlabel('Discharge IOM [Gt/yr]')
plt.ylabel('Partitioned discharge RA [Gt/yr]')

plt.title('Antarctica')

# statistics
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(
    discharge_rignot_basins.astype(float),discharge_ra_cpom_abs.astype(float))
plt.text(340, 75,
         "RA R$^2$ = %.2f" % r_value ** 2,
         ha='right')

plt.text(340, 50,
         "RMSE = %.2f Gt/yr" % mean_squared_error(
             discharge_rignot_basins.astype(float),discharge_ra_cpom_abs.astype(float), squared=False),
         ha='right')

plt.savefig('/Users/thomas/Documents/github/imbie_partitioning/figs/discharge_compare_RA_IOM_ais.svg',
            format='svg', dpi=600, bbox_inches='tight')
