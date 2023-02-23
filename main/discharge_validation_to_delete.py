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
# 1. Antarctica
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
smb_ref_ais = np.load(
    '/Users/thomas/Documents/github/imbie_partitioning/aux/smb_ref/smb_ref_ais.npy')
dmdt_dyn_ra_ais_annual_abs = (smb_ref_ais) - dmdt_dyn_ra_ais_annual
dmdt_dyn_gmb_ais_annual_abs = (smb_ref_ais) - dmdt_dyn_gmb_ais_annual

# adjust for differences in reference smb
# rignot smb_ref_ais = 2020.6 Gt/yr
# dmdt_dyn_ra_ais_annual_abs = dmdt_dyn_ra_ais_annual_abs - \
#     (2020.6 - (smb_ref_ais))
# dmdt_dyn_gmb_ais_annual_abs = dmdt_dyn_gmb_ais_annual_abs - \
#     (2020.6 - (smb_ref_ais))

# load IOM discharge
iom = pd.read_csv('/Users/thomas/Documents/github/imbie_partitioning/aux/iom_datasets/pnas.1812883116.sd01_discharge.csv',
                  float_precision='round_trip')
time_iom = iom['Year'].values
dmdt_dyn_iom_ais = iom['AIS discharge'].values
dmdt_dyn_uncert_iom_ais = iom['AIS discharge uncertainty'].values

# find overlap
time_iom_ra_ais, idx_iom1_ais, idx_ra_ais = np.intersect1d(
    time_iom, time_ra_ais_annual, return_indices=True)
time_iom_gmb_ais, idx_iom2_ais, idx_gmb_ais = np.intersect1d(
    time_iom, time_gmb_ais_annual, return_indices=True)

# compare 5 year rates
# function to compute 5 year averages
def five_year_average(arr):
    result = []
    for i in range(len(arr)):
        # skip if not 5 values
        if i > 1 and i < len(arr) - 2:
            start = max(0, i - 2)
            end = min(len(arr), i + 3)
            group = arr[start:end]
            result.append(np.mean(group))
    return np.array(result, dtype=float)

# # Test the function
# arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# print(five_year_average(arr))

cmap = plt.cm.get_cmap('Set2')

fig = plt.figure(figsize=(4, 4), constrained_layout=True)
gs = plt.GridSpec(1, 1, figure=fig, wspace=0.1)

ax1 = fig.add_subplot(gs[0])

# 1 to 1 line
plt.plot([2000, 2300], [2000, 2300],
         linewidth=0.5,
         color='k')

ax1.scatter(five_year_average(dmdt_dyn_iom_ais[idx_iom1_ais]), five_year_average(dmdt_dyn_ra_ais_annual_abs[idx_ra_ais]),
            color=cmap(0),
            label='RA')

ax1.scatter(five_year_average(dmdt_dyn_iom_ais[idx_iom2_ais]), five_year_average(dmdt_dyn_gmb_ais_annual_abs[idx_gmb_ais]),
            color=cmap(1),
            label='GMB')

ax1.set_xlim((2000, 2300))
ax1.set_ylim((2000, 2300))

ax1.set_aspect('equal', 'box')

plt.xlabel('Discharge IOM [Gt/yr]')
plt.ylabel('Partitioned discharge RA/GMB [Gt/yr]')

plt.legend()

plt.title('Antarctica')

# statistics
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(
    five_year_average(dmdt_dyn_iom_ais[idx_iom1_ais]), five_year_average(dmdt_dyn_ra_ais_annual_abs[idx_ra_ais]))
plt.text(2290, 2070,
         "RA R$^2$ = %.2f" % r_value ** 2,
         ha='right',
         color=cmap(0))

plt.text(2290, 2050,
         "RA RMSE = %.2f Gt/yr" % mean_squared_error(
             five_year_average(dmdt_dyn_iom_ais[idx_iom1_ais]), five_year_average(dmdt_dyn_ra_ais_annual_abs[idx_ra_ais]), squared=False),
         ha='right',
         color=cmap(0))

slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(
    five_year_average(dmdt_dyn_iom_ais[idx_iom2_ais]), five_year_average(dmdt_dyn_gmb_ais_annual_abs[idx_gmb_ais]))
plt.text(2290, 2030,
         "GMB R$^2$ = %.2f" % r_value ** 2,
         ha='right',
         color=cmap(1))

plt.text(2290, 2010,
         "GMB RMSE = %.2f Gt/yr" % mean_squared_error(
             five_year_average(dmdt_dyn_iom_ais[idx_iom2_ais]), five_year_average(dmdt_dyn_gmb_ais_annual_abs[idx_gmb_ais]), squared=False),
         ha='right',
         color=cmap(1))

plt.savefig('/Users/thomas/Documents/github/imbie_partitioning/figs/discharge_compare_5yr_ais.svg',
            format='svg', dpi=600, bbox_inches='tight')
fig.clf()
plt.close(fig)

# =============================================================================
# 2. West Antarctica
# =============================================================================
# load imbie partitioned data
imbie_wais = pd.read_csv('/Users/thomas/Documents/github/imbie_partitioning/partitioned_data/imbie_west_antarctica_2021_Gt_partitioned.csv',
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
dmdt_smb_ra_wais = np.interp(time_ra_wais,
                            time_imbie_wais,
                            dmdt_smb_imbie_wais)

dmdt_dyn_ra_wais = dmdt_ra_wais - dmdt_smb_ra_wais
# compute annual discharge
time_ra_wais_annual = []
dmdt_dyn_ra_wais_annual = []
for i, year in enumerate(np.unique(np.floor(time_ra_wais))):
    dmdts_in_year = dmdt_dyn_ra_wais[np.where(np.floor(time_ra_wais) == year)]
    if len(dmdts_in_year) == 12:
        time_ra_wais_annual.append(year)
        dmdt_dyn_ra_wais_annual.append(dmdts_in_year.sum() / 12)
    del dmdts_in_year

time_ra_wais_annual = np.array(time_ra_wais_annual, dtype=float)
dmdt_dyn_ra_wais_annual = np.array(dmdt_dyn_ra_wais_annual, dtype=float)

# GMB
# interpolate SMB
dmdt_smb_gmb_wais = np.interp(time_gmb_wais,
                             time_imbie_wais,
                             dmdt_smb_imbie_wais)

dmdt_dyn_gmb_wais = dmdt_gmb_wais - dmdt_smb_gmb_wais

# compute annual discharge
time_gmb_wais_annual = []
dmdt_dyn_gmb_wais_annual = []
for i, year in enumerate(np.unique(np.floor(time_gmb_wais))):
    dmdts_in_year = dmdt_dyn_gmb_wais[np.where(np.floor(time_gmb_wais) == year)]
    if len(dmdts_in_year) == 12:
        time_gmb_wais_annual.append(year)
        dmdt_dyn_gmb_wais_annual.append(dmdts_in_year.sum() / 12)
    del dmdts_in_year

time_gmb_wais_annual = np.array(time_gmb_wais_annual, dtype=float)
dmdt_dyn_gmb_wais_annual = np.array(dmdt_dyn_gmb_wais_annual, dtype=float)

# load smb reference period to convert partitioned discharge from anomaly to absolute values
smb_ref_wais = np.load(
    '/Users/thomas/Documents/github/imbie_partitioning/aux/smb_ref/smb_ref_wais.npy')
dmdt_dyn_ra_wais_annual_abs = (smb_ref_wais) - dmdt_dyn_ra_wais_annual
dmdt_dyn_gmb_wais_annual_abs = (smb_ref_wais) - dmdt_dyn_gmb_wais_annual

# adjust for differences in reference smb
# rignot smb_ref_wais = 652.6 Gt/yr
# dmdt_dyn_ra_wais_annual_abs = dmdt_dyn_ra_wais_annual_abs - \
#     (652.6 - (smb_ref_wais))
# dmdt_dyn_gmb_wais_annual_abs = dmdt_dyn_gmb_wais_annual_abs - \
#     (652.6 - (smb_ref_wais))

# load IOM discharge
iom = pd.read_csv('/Users/thomas/Documents/github/imbie_partitioning/aux/iom_datasets/pnas.1812883116.sd01_discharge.csv',
                  float_precision='round_trip')
time_iom = iom['Year'].values
dmdt_dyn_iom_wais = iom['WAIS discharge'].values
dmdt_dyn_uncert_iom_wais = iom['WAIS discharge uncertainty'].values

# find overlap
time_iom_ra_wais, idx_iom1_wais, idx_ra_wais = np.intersect1d(
    time_iom, time_ra_wais_annual, return_indices=True)
time_iom_gmb_wais, idx_iom2_wais, idx_gmb_wais = np.intersect1d(
    time_iom, time_gmb_wais_annual, return_indices=True)

# compare 5 year rates
fig = plt.figure(figsize=(4, 4), constrained_layout=True)
gs = plt.GridSpec(1, 1, figure=fig, wspace=0.1)

ax1 = fig.add_subplot(gs[0])

# # 1 to 1 line
plt.plot([500, 800], [500, 800],
         linewidth=0.5,
         color='k')

ax1.scatter(five_year_average(dmdt_dyn_iom_wais[idx_iom1_wais]), five_year_average(dmdt_dyn_ra_wais_annual_abs[idx_ra_wais]),
            color=cmap(0),
            label='RA')

ax1.scatter(five_year_average(dmdt_dyn_iom_wais[idx_iom2_wais]), five_year_average(dmdt_dyn_gmb_wais_annual_abs[idx_gmb_wais]),
            color=cmap(1),
            label='GMB')

ax1.set_xlim((500, 800))
ax1.set_ylim((500, 800))

ax1.set_aspect('equal', 'box')

plt.xlabel('Discharge IOM [Gt/yr]')
plt.ylabel('Partitioned discharge RA/GMB [Gt/yr]')

plt.legend()

plt.title('West Antarctica')

# statistics
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(
    five_year_average(dmdt_dyn_iom_wais[idx_iom1_wais]), five_year_average(dmdt_dyn_ra_wais_annual_abs[idx_ra_wais]))
plt.text(510, 720,
         "RA R$^2$ = %.2f" % r_value ** 2,
         ha='left',
         color=cmap(0))

plt.text(510, 700,
         "RA RMSE = %.2f Gt/yr" % mean_squared_error(
             five_year_average(dmdt_dyn_iom_wais[idx_iom1_wais]), five_year_average(dmdt_dyn_ra_wais_annual_abs[idx_ra_wais]), squared=False),
         ha='left',
         color=cmap(0))

slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(
    five_year_average(dmdt_dyn_iom_wais[idx_iom2_wais]), five_year_average(dmdt_dyn_gmb_wais_annual_abs[idx_gmb_wais]))
plt.text(510, 680,
         "GMB R$^2$ = %.2f" % r_value ** 2,
         ha='left',
         color=cmap(1))

plt.text(510, 660,
         "GMB RMSE = %.2f Gt/yr" % mean_squared_error(
             five_year_average(dmdt_dyn_iom_wais[idx_iom2_wais]), five_year_average(dmdt_dyn_gmb_wais_annual_abs[idx_gmb_wais]), squared=False),
         ha='left',
         color=cmap(1))

plt.savefig('/Users/thomas/Documents/github/imbie_partitioning/figs/discharge_compare_5yr_wais.svg',
            format='svg', dpi=600, bbox_inches='tight')
fig.clf()
plt.close(fig)

# =============================================================================
# 3. East Antarctica
# =============================================================================
# load imbie partitioned data
imbie_eais = pd.read_csv('/Users/thomas/Documents/github/imbie_partitioning/partitioned_data/imbie_east_antarctica_2021_Gt_partitioned.csv',
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
dmdt_smb_ra_eais = np.interp(time_ra_eais,
                            time_imbie_eais,
                            dmdt_smb_imbie_eais)

dmdt_dyn_ra_eais = dmdt_ra_eais - dmdt_smb_ra_eais
# compute annual discharge
time_ra_eais_annual = []
dmdt_dyn_ra_eais_annual = []
for i, year in enumerate(np.unique(np.floor(time_ra_eais))):
    dmdts_in_year = dmdt_dyn_ra_eais[np.where(np.floor(time_ra_eais) == year)]
    if len(dmdts_in_year) == 12:
        time_ra_eais_annual.append(year)
        dmdt_dyn_ra_eais_annual.append(dmdts_in_year.sum() / 12)
    del dmdts_in_year

time_ra_eais_annual = np.array(time_ra_eais_annual, dtype=float)
dmdt_dyn_ra_eais_annual = np.array(dmdt_dyn_ra_eais_annual, dtype=float)

# GMB
# interpolate SMB
dmdt_smb_gmb_eais = np.interp(time_gmb_eais,
                             time_imbie_eais,
                             dmdt_smb_imbie_eais)

dmdt_dyn_gmb_eais = dmdt_gmb_eais - dmdt_smb_gmb_eais

# compute annual discharge
time_gmb_eais_annual = []
dmdt_dyn_gmb_eais_annual = []
for i, year in enumerate(np.unique(np.floor(time_gmb_eais))):
    dmdts_in_year = dmdt_dyn_gmb_eais[np.where(np.floor(time_gmb_eais) == year)]
    if len(dmdts_in_year) == 12:
        time_gmb_eais_annual.append(year)
        dmdt_dyn_gmb_eais_annual.append(dmdts_in_year.sum() / 12)
    del dmdts_in_year

time_gmb_eais_annual = np.array(time_gmb_eais_annual, dtype=float)
dmdt_dyn_gmb_eais_annual = np.array(dmdt_dyn_gmb_eais_annual, dtype=float)

# load smb reference period to convert partitioned discharge from anomaly to absolute values
smb_ref_eais = np.load(
    '/Users/thomas/Documents/github/imbie_partitioning/aux/smb_ref/smb_ref_eais.npy')
dmdt_dyn_ra_eais_annual_abs = (smb_ref_eais) - dmdt_dyn_ra_eais_annual
dmdt_dyn_gmb_eais_annual_abs = (smb_ref_eais) - dmdt_dyn_gmb_eais_annual

# adjust for differences in reference smb
# rignot smb_ref_eais = 1075 Gt/yr
# dmdt_dyn_ra_eais_annual_abs = dmdt_dyn_ra_eais_annual_abs - \
#     (1075 - (smb_ref_eais))
# dmdt_dyn_gmb_eais_annual_abs = dmdt_dyn_gmb_eais_annual_abs - \
#     (1075 - (smb_ref_eais))

# load IOM discharge
iom = pd.read_csv('/Users/thomas/Documents/github/imbie_partitioning/aux/iom_datasets/pnas.1812883116.sd01_discharge.csv',
                  float_precision='round_trip')
time_iom = iom['Year'].values
dmdt_dyn_iom_eais = iom['EAIS discharge'].values
dmdt_dyn_uncert_iom_eais = iom['EAIS discharge uncertainty'].values

# find overlap
time_iom_ra_eais, idx_iom1_eais, idx_ra_eais = np.intersect1d(
    time_iom, time_ra_eais_annual, return_indices=True)
time_iom_gmb_eais, idx_iom2_eais, idx_gmb_eais = np.intersect1d(
    time_iom, time_gmb_eais_annual, return_indices=True)

# compare 5 year rates
fig = plt.figure(figsize=(4, 4), constrained_layout=True)
gs = plt.GridSpec(1, 1, figure=fig, wspace=0.1)

ax1 = fig.add_subplot(gs[0])

# # 1 to 1 line
plt.plot([900, 1400], [900, 1400],
         linewidth=0.5,
         color='k')

ax1.scatter(five_year_average(dmdt_dyn_iom_eais[idx_iom1_eais]), five_year_average(dmdt_dyn_ra_eais_annual_abs[idx_ra_eais]),
            color=cmap(0),
            label='RA')

ax1.scatter(five_year_average(dmdt_dyn_iom_eais[idx_iom2_eais]), five_year_average(dmdt_dyn_gmb_eais_annual_abs[idx_gmb_eais]),
            color=cmap(1),
            label='GMB')

ax1.set_xlim((900, 1400))
ax1.set_ylim((900, 1400))

ax1.set_aspect('equal', 'box')

plt.xlabel('Discharge IOM [Gt/yr]')
plt.ylabel('Partitioned discharge RA/GMB [Gt/yr]')

plt.legend()

plt.title('East Antarctica')

# statistics
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(
    five_year_average(dmdt_dyn_iom_eais[idx_iom1_eais]), five_year_average(dmdt_dyn_ra_eais_annual_abs[idx_ra_eais]))
plt.text(1390, 1040,
         "RA R$^2$ = %.2f" % r_value ** 2,
         ha='right',
         color=cmap(0))

plt.text(1390, 1000,
         "RA RMSE = %.2f Gt/yr" % mean_squared_error(
             five_year_average(dmdt_dyn_iom_eais[idx_iom1_eais]), five_year_average(dmdt_dyn_ra_eais_annual_abs[idx_ra_eais]), squared=False),
         ha='right',
         color=cmap(0))

slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(
    five_year_average(dmdt_dyn_iom_eais[idx_iom2_eais]), five_year_average(dmdt_dyn_gmb_eais_annual_abs[idx_gmb_eais]))
plt.text(1390, 960,
         "GMB R$^2$ = %.2f" % r_value ** 2,
         ha='right',
         color=cmap(1))

plt.text(1390, 920,
         "GMB RMSE = %.2f Gt/yr" % mean_squared_error(
             five_year_average(dmdt_dyn_iom_eais[idx_iom2_eais]), five_year_average(dmdt_dyn_gmb_eais_annual_abs[idx_gmb_eais]), squared=False),
         ha='right',
         color=cmap(1))

plt.savefig('/Users/thomas/Documents/github/imbie_partitioning/figs/discharge_compare_5yr_eais.svg',
            format='svg', dpi=600, bbox_inches='tight')
fig.clf()
plt.close(fig)

# =============================================================================
# 4. Antarctic Peninsula
# =============================================================================
# load imbie partitioned data
imbie_apis = pd.read_csv('/Users/thomas/Documents/github/imbie_partitioning/partitioned_data/imbie_east_antarctica_2021_Gt_partitioned.csv',
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
dmdt_smb_ra_apis = np.interp(time_ra_apis,
                            time_imbie_apis,
                            dmdt_smb_imbie_apis)

dmdt_dyn_ra_apis = dmdt_ra_apis - dmdt_smb_ra_apis
# compute annual discharge
time_ra_apis_annual = []
dmdt_dyn_ra_apis_annual = []
for i, year in enumerate(np.unique(np.floor(time_ra_apis))):
    dmdts_in_year = dmdt_dyn_ra_apis[np.where(np.floor(time_ra_apis) == year)]
    if len(dmdts_in_year) == 12:
        time_ra_apis_annual.append(year)
        dmdt_dyn_ra_apis_annual.append(dmdts_in_year.sum() / 12)
    del dmdts_in_year

time_ra_apis_annual = np.array(time_ra_apis_annual, dtype=float)
dmdt_dyn_ra_apis_annual = np.array(dmdt_dyn_ra_apis_annual, dtype=float)

# GMB
# interpolate SMB
dmdt_smb_gmb_apis = np.interp(time_gmb_apis,
                             time_imbie_apis,
                             dmdt_smb_imbie_apis)

dmdt_dyn_gmb_apis = dmdt_gmb_apis - dmdt_smb_gmb_apis

# compute annual discharge
time_gmb_apis_annual = []
dmdt_dyn_gmb_apis_annual = []
for i, year in enumerate(np.unique(np.floor(time_gmb_apis))):
    dmdts_in_year = dmdt_dyn_gmb_apis[np.where(np.floor(time_gmb_apis) == year)]
    if len(dmdts_in_year) == 12:
        time_gmb_apis_annual.append(year)
        dmdt_dyn_gmb_apis_annual.append(dmdts_in_year.sum() / 12)
    del dmdts_in_year

time_gmb_apis_annual = np.array(time_gmb_apis_annual, dtype=float)
dmdt_dyn_gmb_apis_annual = np.array(dmdt_dyn_gmb_apis_annual, dtype=float)

# load smb reference period to convert partitioned discharge from anomaly to absolute values
smb_ref_apis = np.load(
    '/Users/thomas/Documents/github/imbie_partitioning/aux/smb_ref/smb_ref_apis.npy')
dmdt_dyn_ra_apis_annual_abs = (smb_ref_apis) - dmdt_dyn_ra_apis_annual
dmdt_dyn_gmb_apis_annual_abs = (smb_ref_apis) - dmdt_dyn_gmb_apis_annual

# adjust for differences in reference smb
# rignot smb_ref_apis = 293 Gt/yr
# dmdt_dyn_ra_apis_annual_abs = dmdt_dyn_ra_apis_annual_abs - \
#     (293 - (smb_ref_apis))
# dmdt_dyn_gmb_apis_annual_abs = dmdt_dyn_gmb_apis_annual_abs - \
#     (293 - (smb_ref_apis))

# load IOM discharge
iom = pd.read_csv('/Users/thomas/Documents/github/imbie_partitioning/aux/iom_datasets/pnas.1812883116.sd01_discharge.csv',
                  float_precision='round_trip')
time_iom = iom['Year'].values
dmdt_dyn_iom_apis = iom['APIS discharge'].values
dmdt_dyn_uncert_iom_apis = iom['APIS discharge uncertainty'].values

# find overlap
time_iom_ra_apis, idx_iom1_apis, idx_ra_apis = np.intersect1d(
    time_iom, time_ra_apis_annual, return_indices=True)
time_iom_gmb_apis, idx_iom2_apis, idx_gmb_apis = np.intersect1d(
    time_iom, time_gmb_apis_annual, return_indices=True)

# compare 5 year rates
fig = plt.figure(figsize=(4, 4), constrained_layout=True)
gs = plt.GridSpec(1, 1, figure=fig, wspace=0.1)

ax1 = fig.add_subplot(gs[0])

# 1 to 1 line
plt.plot([100, 400], [100, 400],
         linewidth=0.5,
         color='k')

ax1.scatter(five_year_average(dmdt_dyn_iom_apis[idx_iom1_apis]), five_year_average(dmdt_dyn_ra_apis_annual_abs[idx_ra_apis]),
            color=cmap(0),
            label='RA')

ax1.scatter(five_year_average(dmdt_dyn_iom_apis[idx_iom2_apis]), five_year_average(dmdt_dyn_gmb_apis_annual_abs[idx_gmb_apis]),
            color=cmap(1),
            label='GMB')

ax1.set_xlim((100, 400))
ax1.set_ylim((100, 400))

ax1.set_aspect('equal', 'box')

plt.xlabel('Discharge IOM [Gt/yr]')
plt.ylabel('Partitioned discharge RA/GMB [Gt/yr]')

plt.legend()

plt.title('Antarctic Peninsula')

# statistics
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(
    five_year_average(dmdt_dyn_iom_apis[idx_iom1_apis]), five_year_average(dmdt_dyn_ra_apis_annual_abs[idx_ra_apis]))
plt.text(110, 340,
         "RA R$^2$ = %.2f" % r_value ** 2,
         ha='left',
         color=cmap(0))

plt.text(110, 320,
         "RA RMSE = %.2f Gt/yr" % mean_squared_error(
             five_year_average(dmdt_dyn_iom_apis[idx_iom1_apis]), five_year_average(dmdt_dyn_ra_apis_annual_abs[idx_ra_apis]), squared=False),
         ha='left',
         color=cmap(0))

slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(
    five_year_average(dmdt_dyn_iom_apis[idx_iom2_apis]), five_year_average(dmdt_dyn_gmb_apis_annual_abs[idx_gmb_apis]))
plt.text(110, 300,
         "GMB R$^2$ = %.2f" % r_value ** 2,
         ha='left',
         color=cmap(1))

plt.text(110, 280,
         "GMB RMSE = %.2f Gt/yr" % mean_squared_error(
             five_year_average(dmdt_dyn_iom_apis[idx_iom2_apis]), five_year_average(dmdt_dyn_gmb_apis_annual_abs[idx_gmb_apis]), squared=False),
         ha='left',
         color=cmap(1))

plt.savefig('/Users/thomas/Documents/github/imbie_partitioning/figs/discharge_compare_5yr_apis.svg',
            format='svg', dpi=600, bbox_inches='tight')
fig.clf()
plt.close(fig)

# =============================================================================
# 5. Greenland
# =============================================================================
# load imbie partitioned data
imbie_gris = pd.read_csv('/Users/thomas/Documents/github/imbie_partitioning/partitioned_data/imbie_greenland_2021_Gt_partitioned.csv',
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
dmdt_smb_ra_gris = np.interp(time_ra_gris,
                            time_imbie_gris,
                            dmdt_smb_imbie_gris)

dmdt_dyn_ra_gris = dmdt_ra_gris - dmdt_smb_ra_gris
# compute annual discharge
time_ra_gris_annual = []
dmdt_dyn_ra_gris_annual = []
for i, year in enumerate(np.unique(np.floor(time_ra_gris))):
    dmdts_in_year = dmdt_dyn_ra_gris[np.where(np.floor(time_ra_gris) == year)]
    if len(dmdts_in_year) == 12:
        time_ra_gris_annual.append(year)
        dmdt_dyn_ra_gris_annual.append(dmdts_in_year.sum() / 12)
    del dmdts_in_year

time_ra_gris_annual = np.array(time_ra_gris_annual, dtype=float)
dmdt_dyn_ra_gris_annual = np.array(dmdt_dyn_ra_gris_annual, dtype=float)

# GMB
# interpolate SMB
dmdt_smb_gmb_gris = np.interp(time_gmb_gris,
                             time_imbie_gris,
                             dmdt_smb_imbie_gris)

dmdt_dyn_gmb_gris = dmdt_gmb_gris - dmdt_smb_gmb_gris

# compute annual discharge
time_gmb_gris_annual = []
dmdt_dyn_gmb_gris_annual = []
for i, year in enumerate(np.unique(np.floor(time_gmb_gris))):
    dmdts_in_year = dmdt_dyn_gmb_gris[np.where(np.floor(time_gmb_gris) == year)]
    if len(dmdts_in_year) == 12:
        time_gmb_gris_annual.append(year)
        dmdt_dyn_gmb_gris_annual.append(dmdts_in_year.sum() / 12)
    del dmdts_in_year

time_gmb_gris_annual = np.array(time_gmb_gris_annual, dtype=float)
dmdt_dyn_gmb_gris_annual = np.array(dmdt_dyn_gmb_gris_annual, dtype=float)

# load smb reference period to convert partitioned discharge from anomaly to absolute values
smb_ref_gris = np.load(
    '/Users/thomas/Documents/github/imbie_partitioning/aux/smb_ref/smb_ref_gris.npy')
dmdt_dyn_ra_gris_annual_abs = (smb_ref_gris) - dmdt_dyn_ra_gris_annual
dmdt_dyn_gmb_gris_annual_abs = (smb_ref_gris) - dmdt_dyn_gmb_gris_annual

# adjust for differences in reference smb
# mouginot smb_ref_gris = 410.6 Gt/yr
# dmdt_dyn_ra_gris_annual_abs = dmdt_dyn_ra_gris_annual_abs - \
#     (410.6 - (smb_ref_gris))
# dmdt_dyn_gmb_gris_annual_abs = dmdt_dyn_gmb_gris_annual_abs - \
#     (410.6 - (smb_ref_gris))

# load IOM discharge
iom = pd.read_csv('/Users/thomas/Documents/github/imbie_partitioning/aux/iom_datasets/pnas.1904242116.sd02_discharge.csv',
                  float_precision='round_trip')
time_iom = iom['Year'].values
dmdt_dyn_iom_gris = iom['GRIS discharge'].values
dmdt_dyn_uncert_iom_gris = iom['GRIS discharge uncertainty'].values

# find overlap
time_iom_ra_gris, idx_iom1_gris, idx_ra_gris = np.intersect1d(
    time_iom, time_ra_gris_annual, return_indices=True)
time_iom_gmb_gris, idx_iom2_gris, idx_gmb_gris = np.intersect1d(
    time_iom, time_gmb_gris_annual, return_indices=True)

# compare 5 year rates
fig = plt.figure(figsize=(4, 4), constrained_layout=True)
gs = plt.GridSpec(1, 1, figure=fig, wspace=0.1)

ax1 = fig.add_subplot(gs[0])

# 1 to 1 line
plt.plot([400, 600], [400, 600],
         linewidth=0.5,
         color='k')

ax1.scatter(five_year_average(dmdt_dyn_iom_gris[idx_iom1_gris]), five_year_average(dmdt_dyn_ra_gris_annual_abs[idx_ra_gris]),
            color=cmap(0),
            label='RA')

ax1.scatter(five_year_average(dmdt_dyn_iom_gris[idx_iom2_gris]), five_year_average(dmdt_dyn_gmb_gris_annual_abs[idx_gmb_gris]),
            color=cmap(1),
            label='GMB')

ax1.set_xlim((400, 600))
ax1.set_ylim((400, 600))

ax1.set_aspect('equal', 'box')

plt.xlabel('Discharge IOM [Gt/yr]')
plt.ylabel('Partitioned discharge RA/GMB [Gt/yr]')

plt.legend()

plt.title('Greenland')

# statistics
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(
    five_year_average(dmdt_dyn_iom_gris[idx_iom1_gris]), five_year_average(dmdt_dyn_ra_gris_annual_abs[idx_ra_gris]))
plt.text(590, 440,
         "RA R$^2$ = %.2f" % r_value ** 2,
         ha='right',
         color=cmap(0))

plt.text(590, 430,
         "RA RMSE = %.2f Gt/yr" % mean_squared_error(
             five_year_average(dmdt_dyn_iom_gris[idx_iom1_gris]), five_year_average(dmdt_dyn_ra_gris_annual_abs[idx_ra_gris]), squared=False),
         ha='right',
         color=cmap(0))

slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(
    five_year_average(dmdt_dyn_iom_gris[idx_iom2_gris]), five_year_average(dmdt_dyn_gmb_gris_annual_abs[idx_gmb_gris]))
plt.text(590, 420,
         "GMB R$^2$ = %.2f" % r_value ** 2,
         ha='right',
         color=cmap(1))

plt.text(590, 410,
         "GMB RMSE = %.2f Gt/yr" % mean_squared_error(
             five_year_average(dmdt_dyn_iom_gris[idx_iom2_gris]), five_year_average(dmdt_dyn_gmb_gris_annual_abs[idx_gmb_gris]), squared=False),
         ha='right',
         color=cmap(1))

plt.savefig('/Users/thomas/Documents/github/imbie_partitioning/figs/discharge_compare_5yr_gris.svg',
            format='svg', dpi=600, bbox_inches='tight')
fig.clf()
plt.close(fig)


