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

import sys
sys.dont_write_bytecode = True
sys.path.append('/Users/thomas/Documents/github/imbie_partitioning/main')

import os
os.chdir('/Users/thomas/Documents/github/imbie_partitioning/main')

import numpy as np

import pandas as pd

import xarray as xr

from datetime import datetime

import matplotlib.pyplot as plt

from dm_to_dmdt import dm_to_dmdt

# =============================================================================
# 1. load SMB datasets
# =============================================================================

# =============================================================================
# Fettweis (Zwally and Rignot, monthly uncertainty )
# =============================================================================
# Zwally
fettweis_zwally = pd.read_csv('/Users/thomas/Documents/github/imbie_partitioning/smb_datasets/Fettweis_surface-mass-balance_626693c921c693.14470954/mass-balance-data/fettweis_GrIS/fettweis_GrIS_Zwally.dat',
                        float_precision = 'round_trip')

# find indices of smb (Zwally region)
fettweis_idx_zwally = np.where(fettweis_zwally['Drainage Region ID'] == 9.9)[0]

time_fettweis = fettweis_zwally['Date (decimal years)'].iloc[fettweis_idx_zwally].values
smb_fettweis_zwally = fettweis_zwally['Relative Mass Change (Gt)'].iloc[fettweis_idx_zwally].values
smb_uncert_fettweis_zwally = fettweis_zwally['Relative Mass Change Uncertainty (Gt)'].iloc[fettweis_idx_zwally].values

# Rignot
fettweis_rignot = pd.read_csv('/Users/thomas/Documents/github/imbie_partitioning/smb_datasets/Fettweis_surface-mass-balance_626693c921c693.14470954/mass-balance-data/fettweis_GrIS/fettweis_GrIS_Rignot2016.dat',
                        float_precision = 'round_trip')

# find indices of smb (Rignot region)
fettweis_idx_rignot = np.where(fettweis_rignot['Drainage Region ID'] == 99)[0]

time_fettweis = fettweis_rignot['Date (decimal years)'].iloc[fettweis_idx_rignot].values
smb_fettweis_rignot = fettweis_rignot['Relative Mass Change (Gt)'].iloc[fettweis_idx_rignot].values
smb_uncert_fettweis_rignot = fettweis_rignot['Relative Mass Change Uncertainty (Gt)'].iloc[fettweis_idx_rignot].values

########################################
# Hansen (Zwally and Rignot, monthly, uncertainty)
########################################
hansen = pd.read_csv('/Users/thomas/Documents/github/imbie_partitioning/smb_datasets/Hansen_surface-mass-balance_626c00297df776.22516405/mass-balance-data/Hansen_SMB_IMBIE3_data.txt',
                     float_precision = 'round_trip')

# find indices of smb (Zwally region)
hansen_idx_zwally = np.where((hansen['Drainage Region Set'] == 'Zwally') & (hansen['Drainage Region ID'] == 'GRIS'))
# monthly time resolution
time_hansen = hansen['Date (decimal years)'].iloc[hansen_idx_zwally].values
smb_hansen_zwally = hansen['Rate of Mass Change (Gt/month)'].iloc[hansen_idx_zwally].values
smb_uncert_hansen_zwally = hansen['Rate of Mass Change Uncertainty (Gt/month)'].iloc[hansen_idx_zwally].values

# find indices of smb (Rignot region)
hansen_idx_rignot = np.where((hansen['Drainage Region Set'] == 'Rignot') & (hansen['Drainage Region ID'] == 'GRIS'))
smb_hansen_rignot = hansen['Rate of Mass Change (Gt/month)'].iloc[hansen_idx_rignot].values
smb_uncert_hansen_rignot = hansen['Rate of Mass Change Uncertainty (Gt/month)'].iloc[hansen_idx_rignot].values

# =============================================================================
# Noel
# =============================================================================
noel = pd.read_csv('/Users/thomas/Documents/github/imbie_partitioning/smb_datasets/Noel_aggregated_smb/ERA5-3H_RACMO2.3p2_GrIS1_IMBIE2.csv',
                   float_precision = 'round_trip')
# monthly time resolution
time_noel = noel['Date (Decimal Years)'].values
smb_noel = noel['Surface Mass Change (Gt)'].values
# use racmo 20% uncertainty
smb_uncert_noel = 0.2 * smb_noel

# =============================================================================
# 2. convert to monthly temporal resolution where needed
# =============================================================================

# =============================================================================
# 3. Interpolate to common monthly time vector
# =============================================================================
time_combined = np.arange(1950 + (1 / 12), 2022 + (1/ 12), 1 / 12)

# create list
smb_list = [smb_fettweis_zwally,
            smb_fettweis_rignot,
            smb_hansen_zwally,
            smb_hansen_rignot,
            smb_noel]

smb_uncert_list = [smb_uncert_fettweis_zwally,
                   smb_uncert_fettweis_rignot,
                   smb_uncert_hansen_zwally,
                   smb_uncert_hansen_rignot,
                   smb_uncert_noel]

time_list = [time_fettweis,
             time_fettweis,
             time_hansen,
             time_hansen,
             time_noel]

# interpolate smb
smb_interp_list = []
for i, smb in enumerate(smb_list):
    smb_interp_tmp = np.interp(time_combined,
                               time_list[i],
                               smb)  
    # mask values where no data
    smb_interp_tmp[(time_combined < time_list[i].min()) | (time_combined > time_list[i].max())] = np.nan
    smb_interp_list.append(smb_interp_tmp)  
    del smb_interp_tmp
    
# interpolate smb uncertainty
smb_uncert_interp_list = []
for i, smb_uncert in enumerate(smb_uncert_list):
    smb_uncert_interp_tmp = np.interp(time_combined,
                               time_list[i],
                               smb_uncert)    
    # mask values where no data
    smb_uncert_interp_tmp[(time_combined < time_list[i].min()) | (time_combined > time_list[i].max())] = np.nan
    smb_uncert_interp_list.append(smb_uncert_interp_tmp)  
    del smb_uncert_interp_tmp
    

# =============================================================================
# 4. Compare SMB datasets
# =============================================================================
# plot
cmap = plt.cm.get_cmap('Paired')

fig = plt.figure(figsize = (7,3),constrained_layout=True)
gs = plt.GridSpec(1, 1, figure=fig, wspace = 0.1)

ax1 = fig.add_subplot(gs[0])

# global plot parameters
line_alpha = 0.5
lw = 1

labels = ['Fettweis Zwally (MAR 3.12.1)',
          'Fettweis Rignot (MAR 3.12.1)',
          'Hansen Rignot (HIRHAM5)',
          'Hansen Zwally (HIRHAM5)',
          'Noël (RACMO 2.3p2)']

cmap_ints = [0, 1, 2, 3, 4]

for i, smb in enumerate(smb_interp_list):
    ax1.plot(time_combined, smb,
             color = cmap(cmap_ints[i]),
             alpha = line_alpha,
             linewidth = lw,
             label = labels[i])


# =============================================================================
# 5. Combine SMB datasets
# =============================================================================
# average monthly SMB data

def average_arrays(array_list):
    stacked_arrays = np.stack(array_list, axis = 0)
    array_average = np.nanmean(stacked_arrays, axis = 0)
    return array_average

smb_combined = average_arrays(smb_interp_list)

# combine uncertainties as root mean square

def combine_smb_uncertainties(array_list):
    stacked_arrays = np.stack(array_list, axis = 0)
    # get count
    num_finite = np.isfinite(stacked_arrays).sum(axis = 0)
    # copmute rms
    array_rms = np.sqrt(np.nanmean(stacked_arrays ** 2, axis = 0))
    return array_rms / np.sqrt(num_finite)

smb_uncert_combined = combine_smb_uncertainties(smb_uncert_interp_list)

# add to plot
ax1.plot(time_combined, smb_combined,
         color = 'k',
         linewidth = lw/2,
         label = 'Combined SMB')

plt.xlabel('Year')
plt.ylabel('Surface Mass Balance [Gt/month]')

ax1.set_ylim(-250, 150)

plt.legend(loc = 'center left',
           bbox_to_anchor=(1, 0.5))

plt.savefig('/Users/thomas/Documents/github/imbie_partitioning/figs/smb_gris_datasets.svg', format = 'svg', dpi = 600, bbox_inches='tight')
fig.clf()
plt.close(fig)

# plot combined smb and uncertainty
fig = plt.figure(figsize = (7,3),constrained_layout=True)
gs = plt.GridSpec(1, 1, figure=fig, wspace = 0.1)

ax1 = fig.add_subplot(gs[0])

ax1.fill_between(time_combined, smb_combined - smb_uncert_combined, smb_combined + smb_uncert_combined,
                 color = 'k',
                 alpha = 0.25,
                 edgecolor = 'none')

ax1.plot(time_combined, smb_combined,
         color = 'k',
         label = 'Combined SMB')

plt.xlabel('Year')
plt.ylabel('Surface Mass Balance [Gt/month]')

ax1.set_ylim(-250, 150)


plt.savefig('/Users/thomas/Documents/github/imbie_partitioning/figs/smb_combined_gris.svg', format = 'svg', dpi = 600, bbox_inches='tight')
fig.clf()
plt.close(fig)

# =============================================================================
# 6. Calculate cumulate anomaly of combined SMB
# =============================================================================

# set reference period
t1_ref, t2_ref = 1960, 1990

smb_ref = smb_combined[(time_combined >= t1_ref) & (time_combined < t2_ref)].mean()

# calculate anomaly
smb_combined_anom = smb_combined - smb_ref

# accumulate anomaly
smb_combined_cumul_anom = smb_combined_anom.cumsum()
    
# smooth
smb_combined_cumul_anom_smoothed = pd.DataFrame(smb_combined_cumul_anom).rolling(window = 36, min_periods = 1, center = True).mean().values.flatten()

# plot cumulative SMB
fig = plt.figure(figsize = (7,3),constrained_layout=True)
gs = plt.GridSpec(1, 1, figure=fig, wspace = 0.1)

ax1 = fig.add_subplot(gs[0])


ax1.plot(time_combined, smb_combined_cumul_anom,
         color = cmap(4),
         alpha = 0.5,
         label = 'Monthly')

ax1.plot(time_combined, smb_combined_cumul_anom_smoothed,
         color = cmap(0),
         label = '36 month smoothed')

ax1.set_ylim(-3500, 500)

plt.xlabel('Year')
plt.ylabel('Cumulative SMB Anomaly [Gt] \n' + '(w.r.t ' + str(t1_ref) + '-' + str(t2_ref) +')')

plt.legend(loc = 'center left',
           bbox_to_anchor=(1, 0.5))

plt.savefig('/Users/thomas/Documents/github/imbie_partitioning/figs/smb_combined_cumulative_anomaly_gris.svg', format = 'svg', dpi = 600, bbox_inches='tight')
fig.clf()
plt.close(fig)

# =============================================================================
# 7. Partition IMBIE mass balance
# =============================================================================
# load IMBIE data
imbie = pd.read_csv('/Users/thomas/Documents/github/imbie_partitioning/imbie_datasets/imbie_greenland_2021_Gt.csv')
time_imbie = imbie['Year'].values
dmdt_imbie = imbie['Mass balance (Gt/yr)'].values

dmdt_uncert_imbie = imbie['Mass balance uncertainty (Gt/yr)'].values

# convert SMB anomaly to dM/dt
dmdt_smb_imbie, num_points = dm_to_dmdt(time_combined, smb_combined_cumul_anom_smoothed, 36)
# need to convert from Gt / month to Gt / yr
dmdt_smb_uncert_imbie = smb_uncert_combined * 12 

# function to find index of nearest input value in a given vector
def dsearchn(x, v):
    return int(np.where(np.abs(x - v) == np.abs(x - v).min())[0])

t1 = dsearchn(np.floor(time_imbie[0]), time_combined)
t2 = dsearchn(np.ceil(time_imbie[-1]), time_combined)

# partition as dynamics = dm - SMB
dmdt_dyn_imbie = dmdt_imbie - dmdt_smb_imbie[t1:t2]
# combine uncertainties
dmdt_dyn_uncert_imbie = np.sqrt(dmdt_uncert_imbie ** 2 + dmdt_smb_uncert_imbie[t1:t2] ** 2)

# integrate for cumulative mass change
dm_imbie = imbie['Cumulative mass balance (Gt)'].values
dm_uncert_imbie = imbie['Cumulative mass balance uncertainty (Gt)'].values

dm_smb_imbie = dmdt_smb_imbie.cumsum() / 12
# accumulate anomaly uncertainty through time (root sum square)
dm_smb_uncert_imbie = np.zeros_like(dm_smb_imbie)
for i, x in enumerate(dm_smb_uncert_imbie):
    if i == 0:
        dm_smb_uncert_imbie[i] = np.sqrt(np.sum(dmdt_smb_uncert_imbie[i] ** 2)) / np.sqrt(12)
    else:
        dm_smb_uncert_imbie[i] = np.sqrt(np.sum(dmdt_smb_uncert_imbie[0:i] ** 2)) / np.sqrt(12)
    
dm_dyn_imbie = dmdt_dyn_imbie.cumsum() / 12
dm_dyn_uncert_imbie = np.zeros_like(dm_dyn_imbie)
for i, x in enumerate(dm_dyn_uncert_imbie):
    if i == 0:
        dm_dyn_uncert_imbie[i] = np.sqrt(np.sum(dmdt_dyn_uncert_imbie[0] ** 2)) / np.sqrt(12)
    else:
        dm_dyn_uncert_imbie[i] = np.sqrt(np.sum(dmdt_dyn_uncert_imbie[0:i] ** 2)) / np.sqrt(12)
    

# =============================================================================
# 8. plot partitioned mass balance
# =============================================================================
# function to offset time series
def time_series_offset(v, offset):
    return v - (v[0] - offset)

offset = dm_smb_imbie[t1]

cmap = plt.cm.get_cmap('PuBuGn', 5)
a = 0.25

fig = plt.figure(figsize = (7,3),constrained_layout=True)
gs = plt.GridSpec(1, 1, figure=fig, wspace = 0.1)

ax1 = fig.add_subplot(gs[0])

# smb
ax1.fill_between(time_combined, 
                 dm_smb_imbie - dm_smb_uncert_imbie, 
                 dm_smb_imbie + dm_smb_uncert_imbie,
                 color = cmap(1),
                 alpha = a,
                 edgecolor = 'none')

ax1.plot(time_combined, dm_smb_imbie,
         color = cmap(1),
         label = 'Surface')

# dynamics
ax1.fill_between(time_imbie, 
                 time_series_offset(dm_dyn_imbie, offset) - dm_dyn_uncert_imbie, 
                 time_series_offset(dm_dyn_imbie, offset) + dm_dyn_uncert_imbie,
                 color = cmap(2),
                 alpha = a,
                 edgecolor = 'none')

ax1.plot(time_imbie, time_series_offset(dm_dyn_imbie, offset),
         color = cmap(2),
         label = 'Dynamics')

# total
ax1.fill_between(time_imbie, 
                 time_series_offset(dm_imbie, offset) - dm_uncert_imbie, 
                 time_series_offset(dm_imbie, offset) + dm_uncert_imbie,
                 color = cmap(3),
                 alpha = a,
                 edgecolor = 'none')

ax1.plot(time_imbie, time_series_offset(dm_imbie, offset),
         color = cmap(3),
         label = 'Total')

plt.xlabel('Year')
plt.ylabel('Mass change [Gt]')

plt.legend(loc = 'center left',
           bbox_to_anchor=(1, 0.5))

plt.xlim(1950, 2021)
plt.ylim(-5000, 1000)

plt.savefig('/Users/thomas/Documents/github/imbie_partitioning/figs/imbie_partitioned_gris.svg', format = 'svg', dpi = 600, bbox_inches='tight')
fig.clf()
plt.close(fig)

# =============================================================================
# 8. Save IMBIE outputs
# =============================================================================

# create dataframe
df_smb = pd.DataFrame({'Year':time_combined[0:t2],
                       'Surface mass balance anomaly (Gt/yr)':dmdt_smb_imbie[0:t2],
                       'Surface mass balance anomaly uncertainty (Gt/yr)':dmdt_smb_uncert_imbie[0:t2],
                       'Cumulative surface mass balance anomaly (Gt)':dm_smb_imbie[0:t2],
                       'Cumulative surface mass balance anomaly uncertainty (Gt)':dm_smb_uncert_imbie[0:t2]})

df_dm = pd.DataFrame({'Mass balance (Gt/yr)':dmdt_imbie,
                      'Mass balance uncertainty (Gt/yr)':dmdt_uncert_imbie,
                      'Cumulative mass balance (Gt)':dm_imbie,
                      'Cumulative mass balance uncertainty (Gt)':dm_uncert_imbie})
# set index for merging
df_dm = df_dm.set_index(np.arange(t1, t2, 1))

df_dyn = pd.DataFrame({'Dynamics mass balance anomaly (Gt/yr)':dmdt_dyn_imbie,
                       'Dynamics mass balance anomaly uncertainty (Gt/yr)':dmdt_dyn_uncert_imbie,
                       'Cumulative dynamics mass balance anomaly (Gt)':dm_dyn_imbie,
                       'Cumulative dynamics mass balance anomaly uncertainty (Gt)':dm_dyn_uncert_imbie})
df_dyn = df_dyn.set_index(np.arange(t1, t2, 1))

# merge dataframes
df_out = pd.merge_asof(df_smb, pd.concat([df_dyn, df_dm], axis = 1),
                      left_index = True,
                      right_index = True)
# remove nans from merging
df_out = df_out.replace({np.nan: None})

# save
df_out.to_csv('/Users/thomas/Documents/github/imbie_partitioning/partitioned_data/imbie_greenland_2021_Gt_partitioned.csv',
              index = False)
