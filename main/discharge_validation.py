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
import scipy.stats

sys.dont_write_bytecode = True
sys.path.append('/Users/thomas/Documents/github/imbie_partitioning/main')

os.chdir('/Users/thomas/Documents/github/imbie_partitioning/main')

# =============================================================================
# 1. AIS
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
dmdt_smb_ra_ais = np.interp(time_ra_ais,
                            time_imbie_ais,
                            dmdt_smb_imbie_ais)

dmdt_dyn_ra_ais = dmdt_ra_ais - dmdt_smb_ra_ais
# compute annual discharge
time_ra_ais_annual = []
dmdt_dyn_ra_ais_annual = []
for i, year in enumerate(np.unique(np.floor(time_ra_ais))):
    dmdts_in_year = dmdt_dyn_ra_ais[np.where(np.floor(time_ra_ais) == year)]
    if len(dmdts_in_year) == 12:
        time_ra_ais_annual.append(year)
        dmdt_dyn_ra_ais_annual.append(dmdts_in_year.sum() / 12)
    del dmdts_in_year

time_ra_ais_annual = np.array(time_ra_ais_annual, dtype=float)
dmdt_dyn_ra_ais_annual = np.array(dmdt_dyn_ra_ais_annual, dtype=float)

# GMB
# interpolate SMB
dmdt_smb_gmb_ais = np.interp(time_gmb_ais,
                             time_imbie_ais,
                             dmdt_smb_imbie_ais)

dmdt_dyn_gmb_ais = dmdt_gmb_ais - dmdt_smb_gmb_ais

# compute annual discharge
time_gmb_ais_annual = []
dmdt_dyn_gmb_ais_annual = []
for i, year in enumerate(np.unique(np.floor(time_gmb_ais))):
    dmdts_in_year = dmdt_dyn_gmb_ais[np.where(np.floor(time_gmb_ais) == year)]
    if len(dmdts_in_year) == 12:
        time_gmb_ais_annual.append(year)
        dmdt_dyn_gmb_ais_annual.append(dmdts_in_year.sum() / 12)
    del dmdts_in_year

time_gmb_ais_annual = np.array(time_gmb_ais_annual, dtype=float)
dmdt_dyn_gmb_ais_annual = np.array(dmdt_dyn_gmb_ais_annual, dtype=float)

# load smb reference period to convert partitioned discharge from anomaly to absolute values
smb_ref = np.load(
    '/Users/thomas/Documents/github/imbie_partitioning/aux/smb_ref/smb_ref_ais.npy')
dmdt_dyn_ra_ais_annual_abs = (smb_ref * 12) - dmdt_dyn_ra_ais_annual
dmdt_dyn_gmb_ais_annual_abs = (smb_ref * 12) - dmdt_dyn_gmb_ais_annual

# adjust for differences in reference smb
dmdt_dyn_ra_ais_annual_abs = dmdt_dyn_ra_ais_annual_abs - \
    (2021 - (smb_ref * 12))
dmdt_dyn_gmb_ais_annual_abs = dmdt_dyn_gmb_ais_annual_abs - \
    (2021 - (smb_ref * 12))

# load IOM discharge
iom = pd.read_csv('/Users/thomas/Documents/github/imbie_partitioning/aux/iom_datasets/pnas.1812883116.sd01_discharge.csv',
                  float_precision='round_trip')
time_iom = iom['Year'].values
dmdt_dyn_iom_ais = iom['AIS discharge'].values
dmdt_dyn_uncert_iom_ais = iom['AIS discharge uncertainty'].values

# scatter plot
cmap = plt.cm.get_cmap('Set2')

fig = plt.figure(figsize=(4, 4), constrained_layout=True)
gs = plt.GridSpec(1, 1, figure=fig, wspace=0.1)

ax1 = fig.add_subplot(gs[0])

# 1 to 1 line
plt.plot([2000, 2300], [2000, 2300],
         linewidth=0.5,
         color='k')

# find overlap
time_ra_gmb, idx_ra, idx_gmb1 = np.intersect1d(
    time_ra_ais_annual, time_gmb_ais_annual, return_indices=True)
time_iom_gmb, idx_iom, idx_gmb2 = np.intersect1d(
    time_iom, time_gmb_ais_annual, return_indices=True)

ax1.scatter(dmdt_dyn_ra_ais_annual_abs[idx_ra], dmdt_dyn_iom_ais[idx_iom],
            color=cmap(0),
            label='RA')

ax1.scatter(dmdt_dyn_gmb_ais_annual_abs[idx_gmb2], dmdt_dyn_iom_ais[idx_iom],
            color=cmap(1),
            label='GMB')

ax1.set_xlim((2100, 2300))
ax1.set_ylim((2100, 2300))

ax1.set_aspect('equal', 'box')

plt.xlabel('Partitioned discharge RA/GMB [Gt/yr]')
plt.ylabel('Discharge IOM [Gt/yr]')

plt.legend()

plt.title('Antarctica')

# statistics
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(
    dmdt_dyn_iom_ais[idx_iom], dmdt_dyn_ra_ais_annual_abs[idx_ra])
plt.text(2290, 2140,
         "RA R$^2$ = %.2f" % r_value ** 2,
         ha='right',
         color=cmap(0))

plt.text(2290, 2130,
         "RA RMSE = %.2f Gt/yr" % mean_squared_error(
             dmdt_dyn_iom_ais[idx_iom], dmdt_dyn_ra_ais_annual_abs[idx_ra], squared=False),
         ha='right',
         color=cmap(0))

slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(
    dmdt_dyn_iom_ais[idx_iom], dmdt_dyn_gmb_ais_annual_abs[idx_gmb2])
plt.text(2290, 2120,
         "GMB R$^2$ = %.2f" % r_value ** 2,
         ha='right',
         color=cmap(1))

plt.text(2290, 2110,
         "GMB RMSE = %.2f Gt/yr" % mean_squared_error(
             dmdt_dyn_iom_ais[idx_iom], dmdt_dyn_gmb_ais_annual_abs[idx_gmb2], squared=False),
         ha='right',
         color=cmap(1))

plt.savefig('/Users/thomas/Documents/github/imbie_partitioning/figs/discharge_compare_ais.svg',
            format='svg', dpi=600, bbox_inches='tight')
fig.clf()
plt.close(fig)
