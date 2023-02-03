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

from dm_to_dmdt import dm_to_dmdt
import matplotlib.pyplot as plt
from datetime import datetime
import xarray as xr
import pandas as pd
import numpy as np
import os
import sys
sys.dont_write_bytecode = True
sys.path.append('/Users/thomas/Documents/github/imbie_partitioning/main')

os.chdir('/Users/thomas/Documents/github/imbie_partitioning/main')

# =============================================================================
# 1. load SMB datasets
# =============================================================================
########################################
# Amory (Zwally only, monthly, uncertainty)
########################################
amory = pd.read_csv('/Users/thomas/Documents/github/imbie_partitioning/aux/smb_datasets/Amory_surface-mass-balance_62b1b713be3696.98550746/mass-balance-data/kittel_amory_agosta_AIS_Zwally.dat',
                    float_precision='round_trip')

# find indices of smb (Zwally region)
# APIS Zwally IDs = 24,25,26,27
amory_apis_idx = ['24', '25', '26', '27']

# monthly time resolution
time_amory = amory['Date (decimal years)'].iloc[np.where(
    amory['Drainage Region ID'] == amory_apis_idx[0])].values

smb_amory_zwally_list = []
smb_uncert_amory_zwally_list = []

# get smb and uncertainty in ice sheet basins
for i, basin in enumerate(amory_apis_idx):
    amory_idx_zwally = np.where(amory['Drainage Region ID'] == basin)
    smb_amory_zwally_list.append(
        amory['Relative Mass Change (Gt)'].iloc[amory_idx_zwally].values)
    smb_uncert_amory_zwally_list.append(
        amory['Relative Mass Change Uncertainty (Gt)'].iloc[amory_idx_zwally].values)

# sum and combine uncertainties
smb_amory_zwally = [sum(x) for x in zip(*smb_amory_zwally_list)]
smb_uncert_amory_zwally = [np.sqrt(sum(x ** 2 for x in row))
                           for row in zip(*smb_uncert_amory_zwally_list)]

########################################
# Hansen (Zwally and Rignot, monthly, uncertainty)
########################################
hansen = pd.read_csv('/Users/thomas/Documents/github/imbie_partitioning/aux/smb_datasets/Hansen_surface-mass-balance_626c00297df776.22516405/mass-balance-data/Hansen_SMB_IMBIE3_data.txt',
                     float_precision='round_trip')

# find indices of smb (Zwally region)
hansen_apis_idx_zwally = ['24.000', '25.000', '26.000', '27.000']

# monthly time resolution
time_hansen = hansen['Date (decimal years)'].iloc[np.where(
    hansen['Drainage Region ID'] == hansen_apis_idx_zwally[0])].values

smb_hansen_zwally_list = []
smb_uncert_hansen_zwally_list = []

# get smb and uncertainty in ice sheet basins
for i, basin in enumerate(hansen_apis_idx_zwally):
    hansen_idx_zwally = np.where(hansen['Drainage Region ID'] == basin)
    smb_hansen_zwally_list.append(
        hansen['Rate of Mass Change (Gt/month)'].iloc[hansen_idx_zwally].values)
    smb_uncert_hansen_zwally_list.append(
        hansen['Rate of Mass Change Uncertainty (Gt/month)'].iloc[hansen_idx_zwally].values)

# sum and combine uncertainties
smb_hansen_zwally = [sum(x) for x in zip(*smb_hansen_zwally_list)]
smb_uncert_hansen_zwally = [np.sqrt(
    sum(x ** 2 for x in row)) for row in zip(*smb_uncert_hansen_zwally_list)]

# find indices of smb (Rignot region)
hansen_apis_idx_rignot = ['Hp-I', 'I-Ipp', 'Ipp-J']

smb_hansen_rignot_list = []
smb_uncert_hansen_rignot_list = []

# get smb and uncertainty in ice sheet basins
for i, basin in enumerate(hansen_apis_idx_rignot):
    hansen_idx_rignot = np.where(hansen['Drainage Region ID'] == basin)
    smb_hansen_rignot_list.append(
        hansen['Rate of Mass Change (Gt/month)'].iloc[hansen_idx_rignot].values)
    smb_uncert_hansen_rignot_list.append(
        hansen['Rate of Mass Change Uncertainty (Gt/month)'].iloc[hansen_idx_rignot].values)

# sum and combine uncertainties
smb_hansen_rignot = [sum(x) for x in zip(*smb_hansen_rignot_list)]
smb_uncert_hansen_rignot = [np.sqrt(
    sum(x ** 2 for x in row)) for row in zip(*smb_uncert_hansen_rignot_list)]

########################################
# Medley (Zwally and Rignot, daily, uncertainty)
########################################
medley = pd.read_csv('/Users/thomas/Documents/github/imbie_partitioning/aux/smb_datasets/Medley_surface-mass-balance_62856337e41344.81177704/mass-balance-data/Medley_AIS_SMB_IMBIE3.csv',
                     float_precision='round_trip')

# find indices of smb (Zwally region)
medley_idx_zwally = np.where((medley['Drainage_Region_Set'] == 'Zwally') & (
    medley['Drainage_Region_ID'] == 'APIS'))
# daily time resolution??
time_medley = medley['Date'].iloc[medley_idx_zwally].values
smb_medley_zwally = medley['Surface_Mass_Balance'].iloc[medley_idx_zwally].values
smb_uncert_medley_zwally = medley['Surface_Mass_Balance_Uncertainty'].iloc[medley_idx_zwally].values

# find indices of smb (Rignot region)
medley_idx_rignot = np.where((medley['Drainage_Region_Set'] == 'Rignot') & (
    medley['Drainage_Region_ID'] == 'APIS'))
smb_medley_rignot = medley['Surface_Mass_Balance'].iloc[medley_idx_rignot].values
smb_uncert_medley_rignot = medley['Surface_Mass_Balance_Uncertainty'].iloc[medley_idx_rignot].values

########################################
# van Wessem (Racmo ice mask, monthly, uncertainty)
########################################

# EAIS Rignot basin numbers = 2,6,7
vwessem_apis_idx = np.array([2, 6, 7]) - 1
ds = xr.open_dataset(
    '/Users/thomas/Documents/github/imbie_partitioning/aux/smb_datasets/van Wessem_surface-mass-balance_6262601bae86e4.66224488/mass-balance-data/ERA5-3H_RACMO2.3p2_ANT27_IMBIE2.nc')
vwessem = pd.DataFrame(np.array([np.arange(1979, 2022, 1/12), ds.smb[:, vwessem_apis_idx].sum(axis=1)]).T,
                       columns=['Time', 'Surface Mass Change'])
del ds
# monthly time resolution
time_vwessem = vwessem['Time'].values
smb_vwessem = vwessem['Surface Mass Change'].values
# use racmo 20% uncertainty
smb_uncert_vwessem = 0.2 * smb_vwessem

# =============================================================================
# 2. convert to monthly temporal resolution where needed
# =============================================================================
########################################
# Medley
########################################
# convert to datetime


def decimal_year_to_datetime(decimal_number_array):
    datetimes_array = []
    for epoch in decimal_number_array:
        # convert text to number
        date_decimal = float(epoch)
        # year is the integer part of the input
        date_year = int(date_decimal)
        # number of days is part of the year, which is left after we subtract year
        year_fraction = date_decimal - date_year
        # a little oversimplified here with int and assuming all years have 365 days
        days = int(year_fraction * 365)
        if days == 0:
            days = days + 1
        # now convert the year and days into string and then into date (there is probably a better way to do this - without the string step)
        date = datetime.strptime("{}-{}".format(date_year, days), "%Y-%j")
        # see https://docs.python.org/3/library/datetime.html#strftime-strptime-behavior for format explanation
        datetimes_array.append(date)
    return datetimes_array


datetime_medley = decimal_year_to_datetime(time_medley)

# resample to monthly
datetime_medley_df = pd.DataFrame({'Datetime': datetime_medley,
                                   'SMB Zwally': smb_medley_zwally,
                                   'SMB Uncertainty Zwally': smb_uncert_medley_zwally,
                                   'SMB Rignot': smb_medley_rignot,
                                   'SMB Uncertainty Rignot': smb_uncert_medley_rignot})

datetime_medley_df.set_index('Datetime',
                             inplace=True)

medley_monthly = datetime_medley_df.resample('M').sum()

# create new numpy arrays
time_medley = np.array(medley_monthly.index.year +
                       (medley_monthly.index.dayofyear-1) / 365, dtype=float)
smb_medley_zwally = medley_monthly['SMB Zwally'].values
smb_uncert_medley_zwally = medley_monthly['SMB Uncertainty Zwally'].values
smb_medley_rignot = medley_monthly['SMB Rignot'].values
smb_uncert_medley_rignot = medley_monthly['SMB Uncertainty Rignot'].values

# =============================================================================
# 3. Interpolate to common monthly time vector
# =============================================================================
time_combined = np.arange(1979, 2022+(1/12), 1/12)

# create list
smb_list = [smb_amory_zwally,
            smb_hansen_zwally,
            smb_hansen_rignot,
            smb_medley_zwally,
            smb_medley_rignot,
            smb_vwessem]

smb_uncert_list = [smb_uncert_amory_zwally,
                   smb_uncert_hansen_zwally,
                   smb_uncert_hansen_rignot,
                   smb_uncert_medley_zwally,
                   smb_uncert_medley_rignot,
                   smb_uncert_vwessem]

time_list = [time_amory,
             time_hansen,
             time_hansen,
             time_medley,
             time_medley,
             time_vwessem]

# interpolate smb
smb_interp_list = []
for i, smb in enumerate(smb_list):
    smb_interp_tmp = np.interp(time_combined,
                               time_list[i],
                               smb)
    # mask values where no data
    smb_interp_tmp[(time_combined < time_list[i].min()) | (
        time_combined > time_list[i].max())] = np.nan
    smb_interp_list.append(smb_interp_tmp)
    del smb_interp_tmp

# interpolate smb uncertainty
smb_uncert_interp_list = []
for i, smb_uncert in enumerate(smb_uncert_list):
    smb_uncert_interp_tmp = np.interp(time_combined,
                                      time_list[i],
                                      smb_uncert)
    # mask values where no data
    smb_uncert_interp_tmp[(time_combined < time_list[i].min()) | (
        time_combined > time_list[i].max())] = np.nan
    smb_uncert_interp_list.append(smb_uncert_interp_tmp)
    del smb_uncert_interp_tmp


# =============================================================================
# 4. Compare SMB datasets
# =============================================================================
# plot
cmap = plt.cm.get_cmap('Paired')

fig = plt.figure(figsize=(7, 3), constrained_layout=True)
gs = plt.GridSpec(1, 1, figure=fig, wspace=0.1)

ax1 = fig.add_subplot(gs[0])

# global plot parameters
line_alpha = 0.5
lw = 1

labels = ['Amory Zwally (MAR 3.12.1)',
          'Hansen Rignot (HIRHAM5)',
          'Hansen Zwally (HIRHAM5)',
          'Medley Rignot (GSFC-FDMv1.2.1)',
          'Medley Zwally (GSFC-FDMv1.2.1)',
          'Van Wessem Rignot (RACMO 2.3p2)']

cmap_ints = [0, 2, 3, 4, 5, 6]

ax1.axhline(y=0,
            color='k',
            linewidth=lw / 2,
            linestyle='--')

for i, smb in enumerate(smb_interp_list):
    ax1.plot(time_combined, smb,
             color=cmap(cmap_ints[i]),
             alpha=line_alpha,
             linewidth=lw,
             label=labels[i])


# =============================================================================
# 5. Combine SMB datasets
# =============================================================================
# average monthly SMB data

def average_arrays(array_list):
    stacked_arrays = np.stack(array_list, axis=0)
    array_average = np.nanmean(stacked_arrays, axis=0)
    return array_average


smb_combined = average_arrays(smb_interp_list)

# combine uncertainties as root mean square


def combine_smb_uncertainties(array_list):
    stacked_arrays = np.stack(array_list, axis=0)
    # get count
    num_finite = np.isfinite(stacked_arrays).sum(axis=0)
    # copmute rms
    array_rms = np.sqrt(np.nanmean(stacked_arrays ** 2, axis=0))
    return array_rms / np.sqrt(num_finite)


smb_uncert_combined = combine_smb_uncertainties(smb_uncert_interp_list)

# add to plot
ax1.plot(time_combined, smb_combined,
         color='k',
         linewidth=lw/2,
         label='Combined SMB')

plt.xlabel('Year')
plt.ylabel('Surface Mass Balance [Gt/month]')

ax1.set_ylim(-250, 350)

plt.legend(loc='center left',
           bbox_to_anchor=(1, 0.5))

plt.title('Antarctic Peninsula')

plt.savefig('/Users/thomas/Documents/github/imbie_partitioning/figs/smb_datasets_apis.svg',
            format='svg', dpi=600, bbox_inches='tight')
fig.clf()
plt.close(fig)

# plot combined smb and uncertainty
fig = plt.figure(figsize=(7, 3), constrained_layout=True)
gs = plt.GridSpec(1, 1, figure=fig, wspace=0.1)

ax1 = fig.add_subplot(gs[0])

ax1.fill_between(time_combined, smb_combined - smb_uncert_combined, smb_combined + smb_uncert_combined,
                 color='k',
                 alpha=0.25,
                 edgecolor='none')

ax1.plot(time_combined, smb_combined,
         color='k',
         label='Combined SMB')

plt.xlabel('Year')
plt.ylabel('Surface Mass Balance [Gt/month]')

ax1.set_ylim(0, 350)

plt.title('Antarctic Peninsula')

plt.savefig('/Users/thomas/Documents/github/imbie_partitioning/figs/smb_combined_apis.svg',
            format='svg', dpi=600, bbox_inches='tight')
fig.clf()
plt.close(fig)

# =============================================================================
# 6. Calculate cumulate anomaly of combined SMB
# =============================================================================

# set reference period
t1_ref, t2_ref = 1979, 2010

smb_ref = smb_combined[(time_combined >= t1_ref) &
                       (time_combined < t2_ref)].mean()

# save reference smb
np.save('/Users/thomas/Documents/github/imbie_partitioning/aux/smb_ref/smb_ref_apis.npy',
        smb_ref)

# calculate anomaly
smb_combined_anom = smb_combined - smb_ref

# accumulate anomaly
smb_combined_cumul_anom = smb_combined_anom.cumsum()

# plot cumulative SMB
fig = plt.figure(figsize=(7, 3), constrained_layout=True)
gs = plt.GridSpec(1, 1, figure=fig, wspace=0.1)

ax1 = fig.add_subplot(gs[0])


ax1.plot(time_combined, smb_combined_cumul_anom,
         color=cmap(4))

ax1.set_ylim(-3000, 750)

plt.xlabel('Year')
plt.ylabel('Cumulative SMB Anomaly [Gt] \n' +
           '(w.r.t ' + str(t1_ref) + '-' + str(t2_ref) + ')')

plt.title('Antarctic Peninsula')

plt.savefig('/Users/thomas/Documents/github/imbie_partitioning/figs/smb_combined_cumulative_anomaly_apis.svg',
            format='svg', dpi=600, bbox_inches='tight')
fig.clf()
plt.close(fig)

# =============================================================================
# 7. Partition IMBIE mass balance
# =============================================================================
# load IMBIE data
imbie = pd.read_csv(
    '/Users/thomas/Documents/github/imbie_partitioning/aux/imbie_datasets/imbie_antarctic_peninsula_2021_Gt.csv')
time_imbie = imbie['Year'].values
dmdt_imbie = imbie['Mass balance (Gt/yr)'].values

dmdt_uncert_imbie = imbie['Mass balance uncertainty (Gt/yr)'].values

# convert SMB anomaly to dM/dt
dmdt_smb_imbie, num_points = dm_to_dmdt(
    time_combined, smb_combined_cumul_anom, 36)
# need to convert from Gt / month to Gt / yr
dmdt_smb_uncert_imbie = smb_uncert_combined * 12

# function to find index of nearest input value in a given vector


def dsearchn(x, v):
    return int(np.where(np.abs(x - v) == np.abs(x - v).min())[0])


t1 = dsearchn(np.floor(time_imbie[0]), time_combined)
t2 = dsearchn(np.ceil(time_imbie[-1]), time_combined)

# partition as dynamics anomaly = dm anomaly - SMB anomaly
dmdt_dyn_imbie = dmdt_imbie - dmdt_smb_imbie[t1:t2]
# combine uncertainties
dmdt_dyn_uncert_imbie = np.sqrt(
    dmdt_uncert_imbie ** 2 + dmdt_smb_uncert_imbie[t1:t2] ** 2)

# integrate for cumulative mass change
dm_imbie = imbie['Cumulative mass balance (Gt)'].values
dm_uncert_imbie = imbie['Cumulative mass balance uncertainty (Gt)'].values

dm_smb_imbie = dmdt_smb_imbie.cumsum() / 12
# accumulate anomaly uncertainty through time (root sum square)
dm_smb_uncert_imbie = np.zeros_like(dm_smb_imbie)
for i, x in enumerate(dm_smb_uncert_imbie):
    if i == 0:
        dm_smb_uncert_imbie[i] = np.sqrt(
            np.sum(dmdt_smb_uncert_imbie[i] ** 2)) / np.sqrt(12)
    else:
        dm_smb_uncert_imbie[i] = np.sqrt(
            np.sum(dmdt_smb_uncert_imbie[0:i] ** 2)) / np.sqrt(12)

dm_dyn_imbie = dmdt_dyn_imbie.cumsum() / 12
dm_dyn_uncert_imbie = np.zeros_like(dm_dyn_imbie)
for i, x in enumerate(dm_dyn_uncert_imbie):
    if i == 0:
        dm_dyn_uncert_imbie[i] = np.sqrt(
            np.sum(dmdt_dyn_uncert_imbie[0] ** 2)) / np.sqrt(12)
    else:
        dm_dyn_uncert_imbie[i] = np.sqrt(
            np.sum(dmdt_dyn_uncert_imbie[0:i] ** 2)) / np.sqrt(12)


# =============================================================================
# 8. plot partitioned mass balance
# =============================================================================
# function to offset time series
def time_series_offset(v, offset):
    return v - (v[0] - offset)


offset = dm_smb_imbie[t1]

cmap = plt.cm.get_cmap('PuBuGn', 5)
a = 0.25

fig = plt.figure(figsize=(7, 3), constrained_layout=True)
gs = plt.GridSpec(1, 1, figure=fig, wspace=0.1)

ax1 = fig.add_subplot(gs[0])

# smb
ax1.fill_between(time_combined,
                 dm_smb_imbie - dm_smb_uncert_imbie,
                 dm_smb_imbie + dm_smb_uncert_imbie,
                 color=cmap(1),
                 alpha=a,
                 edgecolor='none')

ax1.plot(time_combined, dm_smb_imbie,
         color=cmap(1),
         label='Surface')

# dynamics
ax1.fill_between(time_imbie,
                 time_series_offset(dm_dyn_imbie, offset) -
                 dm_dyn_uncert_imbie,
                 time_series_offset(dm_dyn_imbie, offset) +
                 dm_dyn_uncert_imbie,
                 color=cmap(2),
                 alpha=a,
                 edgecolor='none')

ax1.plot(time_imbie, time_series_offset(dm_dyn_imbie, offset),
         color=cmap(2),
         label='Dynamics')

# total
ax1.fill_between(time_imbie,
                 time_series_offset(dm_imbie, offset) - dm_uncert_imbie,
                 time_series_offset(dm_imbie, offset) + dm_uncert_imbie,
                 color=cmap(3),
                 alpha=a,
                 edgecolor='none')

ax1.plot(time_imbie, time_series_offset(dm_imbie, offset),
         color=cmap(3),
         label='Total')

plt.xlabel('Year')
plt.ylabel('Mass change [Gt]')

plt.legend(loc='center left',
           bbox_to_anchor=(1, 0.5))

plt.xlim(1979, 2021)
plt.ylim(-5000, 1000)

plt.title('Antarctic Peninsula')

plt.savefig('/Users/thomas/Documents/github/imbie_partitioning/figs/imbie_partitioned_apis.svg',
            format='svg', dpi=600, bbox_inches='tight')
fig.clf()
plt.close(fig)

# =============================================================================
# 8. Save IMBIE outputs
# =============================================================================

# create dataframe
df_smb = pd.DataFrame({'Year': time_combined[0:t2],
                       'Surface mass balance anomaly (Gt/yr)': dmdt_smb_imbie[0:t2],
                       'Surface mass balance anomaly uncertainty (Gt/yr)': dmdt_smb_uncert_imbie[0:t2],
                       'Cumulative surface mass balance anomaly (Gt)': dm_smb_imbie[0:t2],
                       'Cumulative surface mass balance anomaly uncertainty (Gt)': dm_smb_uncert_imbie[0:t2]})

df_dm = pd.DataFrame({'Mass balance (Gt/yr)': dmdt_imbie,
                      'Mass balance uncertainty (Gt/yr)': dmdt_uncert_imbie,
                      'Cumulative mass balance (Gt)': dm_imbie,
                      'Cumulative mass balance uncertainty (Gt)': dm_uncert_imbie})
# set index for merging
df_dm = df_dm.set_index(np.arange(t1, t2, 1))

df_dyn = pd.DataFrame({'Dynamics mass balance anomaly (Gt/yr)': dmdt_dyn_imbie,
                       'Dynamics mass balance anomaly uncertainty (Gt/yr)': dmdt_dyn_uncert_imbie,
                       'Cumulative dynamics mass balance anomaly (Gt)': dm_dyn_imbie,
                       'Cumulative dynamics mass balance anomaly uncertainty (Gt)': dm_dyn_uncert_imbie})
df_dyn = df_dyn.set_index(np.arange(t1, t2, 1))

# merge dataframes
df_out = pd.merge_asof(df_smb, pd.concat([df_dyn, df_dm], axis=1),
                       left_index=True,
                       right_index=True)
# remove nans from merging
df_out = df_out.replace({np.nan: None})

# save
df_out.to_csv('/Users/thomas/Documents/github/imbie_partitioning/partitioned_data/imbie_antarctic_peninsula_2021_Gt_partitioned.csv',
              index=False)
