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
smb_ref = np.load(
    '/Users/thomas/Documents/github/imbie_partitioning/aux/smb_ref/smb_ref_ais.npy')
dmdt_dyn_ra_ais_annual_abs = (smb_ref * 12) - dmdt_dyn_ra_ais_annual
dmdt_dyn_gmb_ais_annual_abs = (smb_ref * 12) - dmdt_dyn_gmb_ais_annual

# adjust for differences in reference smb
# rignot smb_ref = 2020.6 Gt/yr
dmdt_dyn_ra_ais_annual_abs = dmdt_dyn_ra_ais_annual_abs - \
    (2020.6 - (smb_ref * 12))
dmdt_dyn_gmb_ais_annual_abs = dmdt_dyn_gmb_ais_annual_abs - \
    (2020.6 - (smb_ref * 12))

# load IOM discharge
iom = pd.read_csv('/Users/thomas/Documents/github/imbie_partitioning/aux/iom_datasets/pnas.1812883116.sd01_discharge.csv',
                  float_precision='round_trip')
time_iom = iom['Year'].values
dmdt_dyn_iom_ais = iom['AIS discharge'].values
dmdt_dyn_uncert_iom_ais = iom['AIS discharge uncertainty'].values

# compare annual rates
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
time_iom_ra_ais, idx_iom1_ais, idx_ra_ais = np.intersect1d(
    time_iom, time_ra_ais_annual, return_indices=True)
time_iom_gmb_ais, idx_iom2_ais, idx_gmb_ais = np.intersect1d(
    time_iom, time_gmb_ais_annual, return_indices=True)

ax1.scatter(dmdt_dyn_iom_ais[idx_iom1_ais], dmdt_dyn_ra_ais_annual_abs[idx_ra_ais],
            color=cmap(0),
            label='RA')

ax1.scatter(dmdt_dyn_iom_ais[idx_iom2_ais], dmdt_dyn_gmb_ais_annual_abs[idx_gmb_ais],
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
    dmdt_dyn_iom_ais[idx_iom1_ais], dmdt_dyn_ra_ais_annual_abs[idx_ra_ais])
plt.text(2290, 2070,
         "RA R$^2$ = %.2f" % r_value ** 2,
         ha='right',
         color=cmap(0))

plt.text(2290, 2050,
         "RA RMSE = %.2f Gt/yr" % mean_squared_error(
             dmdt_dyn_iom_ais[idx_iom1_ais], dmdt_dyn_ra_ais_annual_abs[idx_ra_ais], squared=False),
         ha='right',
         color=cmap(0))

slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(
    dmdt_dyn_iom_ais[idx_iom2_ais], dmdt_dyn_gmb_ais_annual_abs[idx_gmb_ais])
plt.text(2290, 2030,
         "GMB R$^2$ = %.2f" % r_value ** 2,
         ha='right',
         color=cmap(1))

plt.text(2290, 2010,
         "GMB RMSE = %.2f Gt/yr" % mean_squared_error(
             dmdt_dyn_iom_ais[idx_iom2_ais], dmdt_dyn_gmb_ais_annual_abs[idx_gmb_ais], squared=False),
         ha='right',
         color=cmap(1))

plt.savefig('/Users/thomas/Documents/github/imbie_partitioning/figs/discharge_compare_annual_ais.svg',
            format='svg', dpi=600, bbox_inches='tight')
fig.clf()
plt.close(fig)

# compare 5 year rates
# function to compute 5 year moving average - CHECK THIS!!!!!!!!!!
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w

fig = plt.figure(figsize=(4, 4), constrained_layout=True)
gs = plt.GridSpec(1, 1, figure=fig, wspace=0.1)

ax1 = fig.add_subplot(gs[0])

# 1 to 1 line
plt.plot([2000, 2300], [2000, 2300],
         linewidth=0.5,
         color='k')

ax1.scatter(moving_average(dmdt_dyn_iom_ais[idx_iom1_ais],5)[2:-2], moving_average(dmdt_dyn_ra_ais_annual_abs[idx_ra_ais], 5)[2:-2],
            color=cmap(0),
            label='RA')

ax1.scatter(moving_average(dmdt_dyn_iom_ais[idx_iom2_ais],5)[2:-2], moving_average(dmdt_dyn_gmb_ais_annual_abs[idx_gmb_ais],5)[2:-2],
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
    moving_average(dmdt_dyn_iom_ais[idx_iom1_ais],5)[2:-2], moving_average(dmdt_dyn_ra_ais_annual_abs[idx_ra_ais], 5)[2:-2])
plt.text(2290, 2070,
         "RA R$^2$ = %.2f" % r_value ** 2,
         ha='right',
         color=cmap(0))

plt.text(2290, 2050,
         "RA RMSE = %.2f Gt/yr" % mean_squared_error(
             moving_average(dmdt_dyn_iom_ais[idx_iom1_ais],5)[2:-2], moving_average(dmdt_dyn_ra_ais_annual_abs[idx_ra_ais], 5)[2:-2], squared=False),
         ha='right',
         color=cmap(0))

slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(
    moving_average(dmdt_dyn_iom_ais[idx_iom2_ais],5)[2:-2], moving_average(dmdt_dyn_gmb_ais_annual_abs[idx_gmb_ais],5)[2:-2])
plt.text(2290, 2030,
         "GMB R$^2$ = %.2f" % r_value ** 2,
         ha='right',
         color=cmap(1))

plt.text(2290, 2010,
         "GMB RMSE = %.2f Gt/yr" % mean_squared_error(
             moving_average(dmdt_dyn_iom_ais[idx_iom2_ais],5)[2:-2], moving_average(dmdt_dyn_gmb_ais_annual_abs[idx_gmb_ais],5)[2:-2], squared=False),
         ha='right',
         color=cmap(1))

plt.savefig('/Users/thomas/Documents/github/imbie_partitioning/figs/discharge_compare_5yr_ais.svg',
            format='svg', dpi=600, bbox_inches='tight')
fig.clf()
plt.close(fig)

# =============================================================================
# X. Greenland
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
smb_ref = np.load(
    '/Users/thomas/Documents/github/imbie_partitioning/aux/smb_ref/smb_ref_gris.npy')
dmdt_dyn_ra_gris_annual_abs = (smb_ref) - dmdt_dyn_ra_gris_annual
dmdt_dyn_gmb_gris_annual_abs = (smb_ref) - dmdt_dyn_gmb_gris_annual

# adjust for differences in reference smb
# mouginot smb_ref = 410.6 Gt/yr
dmdt_dyn_ra_gris_annual_abs = dmdt_dyn_ra_gris_annual_abs - \
    (410.6 - (smb_ref))
dmdt_dyn_gmb_gris_annual_abs = dmdt_dyn_gmb_gris_annual_abs - \
    (410.6 - (smb_ref))

# load IOM discharge
iom = pd.read_csv('/Users/thomas/Documents/github/imbie_partitioning/aux/iom_datasets/pnas.1904242116.sd02_discharge.csv',
                  float_precision='round_trip')
time_iom = iom['Year'].values
dmdt_dyn_iom_gris = iom['GRIS discharge'].values
dmdt_dyn_uncert_iom_gris = iom['GRIS discharge uncertainty'].values

# compare annual rates
# scatter plot
cmap = plt.cm.get_cmap('Set2')

fig = plt.figure(figsize=(4, 4), constrained_layout=True)
gs = plt.GridSpec(1, 1, figure=fig, wspace=0.1)

ax1 = fig.add_subplot(gs[0])

# 1 to 1 line
plt.plot([400, 600], [400, 600],
         linewidth=0.5,
         color='k')

# find overlap
time_iom_ra_gris, idx_iom1_gris, idx_ra_gris = np.intersect1d(
    time_iom, time_ra_gris_annual, return_indices=True)
time_iom_gmb_gris, idx_iom2_gris, idx_gmb_gris = np.intersect1d(
    time_iom, time_gmb_gris_annual, return_indices=True)

ax1.scatter(dmdt_dyn_iom_gris[idx_iom1_gris], dmdt_dyn_ra_gris_annual_abs[idx_ra_gris],
            color=cmap(0),
            label='RA')

ax1.scatter(dmdt_dyn_iom_gris[idx_iom2_gris], dmdt_dyn_gmb_gris_annual_abs[idx_gmb_gris],
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
    dmdt_dyn_iom_gris[idx_iom1_gris], dmdt_dyn_ra_gris_annual_abs[idx_ra_gris])
plt.text(590, 440,
         "RA R$^2$ = %.2f" % r_value ** 2,
         ha='right',
         color=cmap(0))

plt.text(590, 430,
         "RA RMSE = %.2f Gt/yr" % mean_squared_error(
             dmdt_dyn_iom_gris[idx_iom1_gris], dmdt_dyn_ra_gris_annual_abs[idx_ra_gris], squared=False),
         ha='right',
         color=cmap(0))

slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(
    dmdt_dyn_iom_gris[idx_iom2_gris], dmdt_dyn_gmb_gris_annual_abs[idx_gmb_gris])
plt.text(590, 420,
         "GMB R$^2$ = %.2f" % r_value ** 2,
         ha='right',
         color=cmap(1))

plt.text(590, 410,
         "GMB RMSE = %.2f Gt/yr" % mean_squared_error(
             dmdt_dyn_iom_gris[idx_iom2_gris], dmdt_dyn_gmb_gris_annual_abs[idx_gmb_gris], squared=False),
         ha='right',
         color=cmap(1))

plt.savefig('/Users/thomas/Documents/github/imbie_partitioning/figs/discharge_compare_annual_gris.svg',
            format='svg', dpi=600, bbox_inches='tight')
fig.clf()
plt.close(fig)

# compare 5 year rates
fig = plt.figure(figsize=(4, 4), constrained_layout=True)
gs = plt.GridSpec(1, 1, figure=fig, wspace=0.1)

ax1 = fig.add_subplot(gs[0])

# 1 to 1 line
plt.plot([400, 600], [400, 600],
         linewidth=0.5,
         color='k')

ax1.scatter(moving_average(dmdt_dyn_iom_gris[idx_iom1_gris],5)[2:-2], moving_average(dmdt_dyn_ra_gris_annual_abs[idx_ra_gris], 5)[2:-2],
            color=cmap(0),
            label='RA')

ax1.scatter(moving_average(dmdt_dyn_iom_gris[idx_iom2_gris],5)[2:-2], moving_average(dmdt_dyn_gmb_gris_annual_abs[idx_gmb_gris],5)[2:-2],
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
    moving_average(dmdt_dyn_iom_gris[idx_iom1_gris],5)[2:-2], moving_average(dmdt_dyn_ra_gris_annual_abs[idx_ra_gris], 5)[2:-2])
plt.text(590, 440,
         "RA R$^2$ = %.2f" % r_value ** 2,
         ha='right',
         color=cmap(0))

plt.text(590, 430,
         "RA RMSE = %.2f Gt/yr" % mean_squared_error(
             moving_average(dmdt_dyn_iom_gris[idx_iom1_gris],5)[2:-2], moving_average(dmdt_dyn_ra_gris_annual_abs[idx_ra_gris], 5)[2:-2], squared=False),
         ha='right',
         color=cmap(0))

slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(
    moving_average(dmdt_dyn_iom_gris[idx_iom2_gris],5)[2:-2], moving_average(dmdt_dyn_gmb_gris_annual_abs[idx_gmb_gris],5)[2:-2])
plt.text(590, 420,
         "GMB R$^2$ = %.2f" % r_value ** 2,
         ha='right',
         color=cmap(1))

plt.text(590, 410,
         "GMB RMSE = %.2f Gt/yr" % mean_squared_error(
             moving_average(dmdt_dyn_iom_gris[idx_iom2_gris],5)[2:-2], moving_average(dmdt_dyn_gmb_gris_annual_abs[idx_gmb_gris],5)[2:-2], squared=False),
         ha='right',
         color=cmap(1))

plt.savefig('/Users/thomas/Documents/github/imbie_partitioning/figs/discharge_compare_5yr_gris.svg',
            format='svg', dpi=600, bbox_inches='tight')
fig.clf()
plt.close(fig)