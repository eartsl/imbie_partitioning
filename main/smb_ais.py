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
########################################
# Amory (Zwally only, monthly, uncertainty)
########################################
amory = pd.read_csv('/Users/thomas/Documents/github/imbie_partitioning/smb_datasets/Amory_surface-mass-balance_62b1b713be3696.98550746/mass-balance-data/kittel_amory_agosta_AIS_Zwally.dat',
                    float_precision = 'round_trip')

# find indices of smb (Zwally region)
amory_idx_zwally = np.where(amory['Drainage Region ID'] == 'ais')
# monthly time resolution
time_amory = amory['Date (decimal years)'].iloc[amory_idx_zwally].values
smb_amory_zwally = amory['Relative Mass Change (Gt)'].iloc[amory_idx_zwally].values
smb_uncert_amory_zwally = amory['Relative Mass Change Uncertainty (Gt)'].iloc[amory_idx_zwally].values

########################################
# Hansen (Zwally and Rignot, monthly, uncertainty)
########################################
hansen = pd.read_csv('/Users/thomas/Documents/github/imbie_partitioning/smb_datasets/Hansen_surface-mass-balance_626c00297df776.22516405/mass-balance-data/Hansen_SMB_IMBIE3_data.txt',
                     float_precision = 'round_trip')

# find indices of smb (Zwally region)
hansen_idx_zwally = np.where((hansen['Drainage Region Set'] == 'Zwally') & (hansen['Drainage Region ID'] == 'AIS'))
# monthly time resolution
time_hansen = hansen['Date (decimal years)'].iloc[hansen_idx_zwally].values
smb_hansen_zwally = hansen['Rate of Mass Change (Gt/month)'].iloc[hansen_idx_zwally].values
smb_uncert_hansen_zwally = hansen['Rate of Mass Change Uncertainty (Gt/month)'].iloc[hansen_idx_zwally].values

# find indices of smb (Rignot region)
hansen_idx_rignot = np.where((hansen['Drainage Region Set'] == 'Rignot') & (hansen['Drainage Region ID'] == 'AIS'))
smb_hansen_rignot = hansen['Rate of Mass Change (Gt/month)'].iloc[hansen_idx_rignot].values
smb_uncert_hansen_rignot = hansen['Rate of Mass Change Uncertainty (Gt/month)'].iloc[hansen_idx_rignot].values

########################################
# Medley (Zwally and Rignot, daily, uncertainty)
########################################
medley = pd.read_csv('/Users/thomas/Documents/github/imbie_partitioning/smb_datasets/Medley_surface-mass-balance_62856337e41344.81177704/mass-balance-data/Medley_AIS_SMB_IMBIE3.csv',
                     float_precision = 'round_trip')

# find indices of smb (Zwally region)
medley_idx_zwally = np.where((medley['Drainage_Region_Set'] == 'Zwally') & (medley['Drainage_Region_ID'] == 'AIS'))
# daily time resolution??
time_medley = medley['Date'].iloc[medley_idx_zwally].values
smb_medley_zwally = medley['Surface_Mass_Balance'].iloc[medley_idx_zwally].values
smb_uncert_medley_zwally = medley['Surface_Mass_Balance_Uncertainty'].iloc[medley_idx_zwally].values

# find indices of smb (Rignot region)
medley_idx_rignot = np.where((medley['Drainage_Region_Set'] == 'Rignot') & (medley['Drainage_Region_ID'] == 'AIS'))
smb_medley_rignot = medley['Surface_Mass_Balance'].iloc[medley_idx_rignot].values
smb_uncert_medley_rignot = medley['Surface_Mass_Balance_Uncertainty'].iloc[medley_idx_rignot].values

"""
########################################
# Niwano (Zwally and Rignot, annual, uncertainty)
########################################
niwano = pd.read_csv('/Users/thomas/Documents/github/imbie_partitioning/smb_datasets/Niwano_surface-mass-balance_62578ab94d14f6.50682970/mass-balance-data/NHM-SMAP_v1.0_1979-2021_rate-of-mass-change_IMBIE3_AIS.csv',
                     float_precision = 'round_trip')

# find indices of smb (Zwally region)
niwano_idx_zwally = np.where((niwano['Drainage-Region-Set'] == 'Zwally') & (niwano['Drainage-Region-ID'] == 'AIS'))
# annual time resolution
time_niwano_start = niwano['Start-Date-(decimal-years)'].iloc[niwano_idx_zwally].values
time_niwano_end = niwano['End-Date-(decimal-years)'].iloc[niwano_idx_zwally].values

smb_niwano_zwally = niwano['Rate-of-Mass-Change-(Gt/yr)'].iloc[niwano_idx_zwally].values
smb_uncert_niwano_zwally = niwano['Rate-of-Mass-Change-Uncertainty-(Gt/yr)'].iloc[niwano_idx_zwally].values

# find indices of smb (Rignot region)
niwano_idx_rignot = np.where((niwano['Drainage-Region-Set'] == 'Rignot') & (niwano['Drainage-Region-ID'] == ' AIS'))
smb_niwano_rignot = niwano['Rate-of-Mass-Change-(Gt/yr)'].iloc[niwano_idx_rignot].values
smb_uncert_niwano_rignot = niwano['Rate-of-Mass-Change-Uncertainty-(Gt/yr)'].iloc[niwano_idx_rignot].values
"""

########################################
# van Wessem (Racmo ice mask, monthly, uncertainty)
########################################
ds = xr.open_dataset('/Users/thomas/Documents/github/imbie_partitioning/smb_datasets/van Wessem_surface-mass-balance_6262601bae86e4.66224488/mass-balance-data/ERA5-3H_RACMO2.3p2_ANT27_IMBIE2.nc')
vwessem = pd.DataFrame(np.array([np.arange(1979,2022,1/12), ds.smb[:,-1]]).T,
                       columns = ['Time', 'Surface Mass Change'])
del ds
# monthly time resolution
time_vwessem = vwessem['Time'].values
smb_vwessem = vwessem['Surface Mass Change'].values
# use racmo 20% uncertainty
smb_uncert_vwessem = 0.2 * smb_vwessem

"""

########################################
# Wever (Zwally and Rignot, daily, no uncertainty)
########################################
wever = pd.read_csv('/Users/thomas/Documents/github/imbie_partitioning/smb_datasets/Wever_surface-mass-balance_6268d0772b5c72.42340985/mass-balance-data/imbie.csv',
                    skiprows =  np.arange(107388, 1232037, 1),
                    float_precision = 'round_trip')

# need to sum AP, EAIS, WAIS
# Zwally
# find indices of WAIS smb (Zwally region)
wever_idx_zwally_wais = np.where((wever['Drainage Region Set (Rignot/Zwally)'] == 'Zwally') & (wever['Drainage Region ID'] == 'WAIS'))
# daily time resolution?
time_wever = wever['Date (decimal years)'].iloc[wever_idx_zwally_wais].values
smb_wever_wais_zwally = wever['Relative Mass Change (Gt)'].iloc[wever_idx_zwally_wais].values

# find indices of EAIS smb (Zwally region)
wever_idx_zwally_eais = np.where((wever['Drainage Region Set (Rignot/Zwally)'] == 'Zwally') & (wever['Drainage Region ID'] == 'EAIS'))
smb_wever_eais_zwally = wever['Relative Mass Change (Gt)'].iloc[wever_idx_zwally_eais].values

# find indices of APIS smb (Zwally region)
wever_idx_zwally_ap = np.where((wever['Drainage Region Set (Rignot/Zwally)'] == 'Zwally') & (wever['Drainage Region ID'] == 'AP'))
smb_wever_ap_zwally = wever['Relative Mass Change (Gt)'].iloc[wever_idx_zwally_ap].values

# sum
smb_wever_zwally = smb_wever_wais_zwally + smb_wever_eais_zwally + smb_wever_ap_zwally

# Rignot
# find indices of WAIS smb (Rignot region)
wever_idx_rignot_wais = np.where((wever['Drainage Region Set (Rignot/Zwally)'] == 'Rignot') & (wever['Drainage Region ID'] == 'West'))
# daily time resolution?
time_wever = wever['Date (decimal years)'].iloc[wever_idx_rignot_wais].values
smb_wever_wais_rignot = wever['Relative Mass Change (Gt)'].iloc[wever_idx_rignot_wais].values

# find indices of EAIS smb (Rignot region)
wever_idx_rignot_eais = np.where((wever['Drainage Region Set (Rignot/Zwally)'] == 'Rignot') & (wever['Drainage Region ID'] == 'East'))
smb_wever_eais_rignot = wever['Relative Mass Change (Gt)'].iloc[wever_idx_rignot_eais].values

# find indices of APIS smb (Rignot region)
wever_idx_rignot_ap = np.where((wever['Drainage Region Set (Rignot/Zwally)'] == 'Rignot') & (wever['Drainage Region ID'] == 'Peninsula'))
smb_wever_ap_rignot = wever['Relative Mass Change (Gt)'].iloc[wever_idx_rignot_ap].values

# sum
smb_wever_rignot = smb_wever_wais_rignot + smb_wever_eais_rignot + smb_wever_ap_rignot
"""

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
        date = datetime.strptime("{}-{}".format(date_year, days),"%Y-%j")  
        # see https://docs.python.org/3/library/datetime.html#strftime-strptime-behavior for format explanation
        datetimes_array.append(date)        
    return datetimes_array
    
datetime_medley = decimal_year_to_datetime(time_medley)

# resample to monthly
datetime_medley_df = pd.DataFrame({'Datetime':datetime_medley,
                                   'SMB Zwally':smb_medley_zwally,
                                   'SMB Uncertainty Zwally':smb_uncert_medley_zwally,
                                   'SMB Rignot':smb_medley_rignot,
                                   'SMB Uncertainty Rignot':smb_uncert_medley_rignot})   

datetime_medley_df.set_index('Datetime',
                             inplace = True)

medley_monthly = datetime_medley_df.resample('M').sum()

# create new numpy arrays
time_medley = np.array(medley_monthly.index.year + (medley_monthly.index.dayofyear-1) / 365, dtype = float)
smb_medley_zwally = medley_monthly['SMB Zwally'].values
smb_uncert_medley_zwally = medley_monthly['SMB Uncertainty Zwally'].values
smb_medley_rignot = medley_monthly['SMB Rignot'].values
smb_uncert_medley_rignot = medley_monthly['SMB Uncertainty Rignot'].values

"""
########################################
# Niwano
########################################
datetime_niwano_start = decimal_year_to_datetime(time_niwano_start)
datetime_niwano_end = decimal_year_to_datetime(time_niwano_end)

# resample to monthly
datetime_niwano_df = pd.DataFrame({'Datetime Start':datetime_niwano_start,
                                   'Datetime End':datetime_niwano_end,
                                   'SMB Zwally':smb_niwano_zwally,
                                   'SMB Uncertainty Zwally':smb_uncert_niwano_zwally,
                                   'SMB Rignot':smb_niwano_rignot,
                                   'SMB Uncertainty Rignot':smb_uncert_niwano_rignot})   

datetime_niwano_df['Date'] = datetime_niwano_df.apply(lambda x: pd.date_range(x['Datetime Start'], x['Datetime End'], freq = 'M'), 
                                                      axis=1)

niwano_monthly = (datetime_niwano_df.explode('Date', ignore_index=True).drop(columns=['Datetime Start', 'Datetime End']))

niwano_monthly.set_index('Date',
                         inplace = True)
# create new numpy arrays
time_niwano = np.array(niwano_monthly.index.year + (niwano_monthly.index.dayofyear-1) / 365, dtype = float)
smb_niwano_zwally = niwano_monthly['SMB Zwally'].values / 12
smb_uncert_niwano_zwally = niwano_monthly['SMB Uncertainty Zwally'].values / 12
smb_niwano_rignot = niwano_monthly['SMB Rignot'].values / 12
smb_uncert_niwano_rignot = niwano_monthly['SMB Uncertainty Rignot'].values / 12
"""
"""
########################################
# Wever
########################################
datetime_wever = decimal_year_to_datetime(time_wever)

# resample to monthly
datetime_wever_df = pd.DataFrame({'Datetime':datetime_wever,
                                   'SMB Zwally':smb_wever_zwally,                               
                                   'SMB Rignot':smb_wever_rignot})   

datetime_wever_df.set_index('Datetime',
                             inplace = True)

wever_monthly = datetime_wever_df.resample('M').sum()

# create new numpy arrays
time_wever = np.array(wever_monthly.index.year + (wever_monthly.index.dayofyear-1) / 365, dtype = float)
smb_wever_zwally = wever_monthly['SMB Zwally'].values
smb_wever_rignot = wever_monthly['SMB Rignot'].values
"""
# =============================================================================
# 3. Interpolate to common monthly time vector
# =============================================================================
time_combined = np.arange(1979,2022+(1/12),1/12)

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

labels = ['Amory Zwally (MAR 3.12.1)',
          'Hansen Rignot (HIRHAM5)',
          'Hansen Zwally (HIRHAM5)',
          'Medley Rignot (MERRA-2)',
          'Medley Zwally (MERRA-2)',
          'Van Wessem (RACMO 2.3p2)']

cmap_ints = [0, 2, 3, 4, 5, 6]

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

ax1.set_ylim(0, 350)

plt.legend(loc = 'center left',
           bbox_to_anchor=(1, 0.5))

plt.savefig('/Users/thomas/Documents/github/imbie_partitioning/figs/smb_datasets_ais.svg', format = 'svg', dpi = 600, bbox_inches='tight')
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

ax1.set_ylim(0, 350)


plt.savefig('/Users/thomas/Documents/github/imbie_partitioning/figs/smb_combined_ais.svg', format = 'svg', dpi = 600, bbox_inches='tight')
fig.clf()
plt.close(fig)

# =============================================================================
# 6. Calculate cumulate anomaly of combined SMB
# =============================================================================

# set reference period
t1_ref, t2_ref = 1979, 2010

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

ax1.set_ylim(-1000, 1000)

plt.xlabel('Year')
plt.ylabel('Cumulative SMB Anomaly [Gt] \n' + '(w.r.t ' + str(t1_ref) + '-' + str(t2_ref) +')')

plt.legend(loc = 'center left',
           bbox_to_anchor=(1, 0.5))

plt.savefig('/Users/thomas/Documents/github/imbie_partitioning/figs/smb_combined_cumulative_anomaly_ais.svg', format = 'svg', dpi = 600, bbox_inches='tight')
fig.clf()
plt.close(fig)

# =============================================================================
# 7. Partition IMBIE mass balance
# =============================================================================
# load IMBIE data
imbie = pd.read_csv('/Users/thomas/Documents/github/imbie_partitioning/imbie_datasets/imbie_antarctica_2021_Gt.csv')
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

plt.xlim(1979, 2021)
plt.ylim(-5000, 1000)

plt.savefig('/Users/thomas/Documents/github/imbie_partitioning/figs/imbie_partitioned_ais.svg', format = 'svg', dpi = 600, bbox_inches='tight')
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
df_out.to_csv('/Users/thomas/Documents/github/imbie_partitioning/partitioned_data/imbie_antarctica_2021_Gt_partitioned.csv',
              index = False)
