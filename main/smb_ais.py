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

# =============================================================================
# 1. load AIS SMB datasets
# =============================================================================
########################################
# Amory (Zwally only, monthly, uncertainty)
########################################
amory = pd.read_csv('/Users/thomas/Documents/github/imbie_partitioning/smb_datasets/Amory_surface-mass-balance_62b1b713be3696.98550746/mass-balance-data/kittel_amory_agosta_AIS_Zwally.dat',
                    float_precision = 'round_trip')

# find indices of AIS smb (Zwally region)
amory_idx_zwally = np.where(amory['Drainage Region ID'] == 'ais')
# monthly time resolution
time_amory = amory['Date (decimal years)'].iloc[amory_idx_zwally].values
smb_amory_ais_zwally = amory['Relative Mass Change (Gt)'].iloc[amory_idx_zwally].values
smb_uncert_amory_ais_zwally = amory['Relative Mass Change Uncertainty (Gt)'].iloc[amory_idx_zwally].values

########################################
# Hansen (Zwally and Rignot, monthly, uncertainty)
########################################
hansen = pd.read_csv('/Users/thomas/Documents/github/imbie_partitioning/smb_datasets/Hansen_surface-mass-balance_626c00297df776.22516405/mass-balance-data/Hansen_SMB_IMBIE3_data.txt',
                     float_precision = 'round_trip')

# find indices of AIS smb (Zwally region)
hansen_idx_zwally = np.where((hansen['Drainage Region Set'] == 'Zwally') & (hansen['Drainage Region ID'] == 'AIS'))
# monthly time resolution
time_hansen = hansen['Date (decimal years)'].iloc[hansen_idx_zwally].values
smb_hansen_ais_zwally = hansen['Rate of Mass Change (Gt/month)'].iloc[hansen_idx_zwally].values
smb_uncert_hansen_ais_zwally = hansen['Rate of Mass Change Uncertainty (Gt/month)'].iloc[hansen_idx_zwally].values

# find indices of AIS smb (Rignot region)
hansen_idx_rignot = np.where((hansen['Drainage Region Set'] == 'Rignot') & (hansen['Drainage Region ID'] == 'AIS'))
smb_hansen_ais_rignot = hansen['Rate of Mass Change (Gt/month)'].iloc[hansen_idx_rignot].values
smb_uncert_hansen_ais_rignot = hansen['Rate of Mass Change Uncertainty (Gt/month)'].iloc[hansen_idx_rignot].values

########################################
# Medley (Zwally and Rignot, daily, uncertainty)
########################################
medley = pd.read_csv('/Users/thomas/Documents/github/imbie_partitioning/smb_datasets/Medley_surface-mass-balance_62856337e41344.81177704/mass-balance-data/Medley_AIS_SMB_IMBIE3.csv',
                     float_precision = 'round_trip')

# find indices of AIS smb (Zwally region)
medley_idx_zwally = np.where((medley['Drainage_Region_Set'] == 'Zwally') & (medley['Drainage_Region_ID'] == 'AIS'))
# daily time resolution??
time_medley = medley['Date'].iloc[medley_idx_zwally].values
smb_medley_ais_zwally = medley['Surface_Mass_Balance'].iloc[medley_idx_zwally].values
smb_uncert_medley_ais_zwally = medley['Surface_Mass_Balance_Uncertainty'].iloc[medley_idx_zwally].values

# find indices of AIS smb (Rignot region)
medley_idx_rignot = np.where((medley['Drainage_Region_Set'] == 'Rignot') & (medley['Drainage_Region_ID'] == 'AIS'))
smb_medley_ais_rignot = medley['Surface_Mass_Balance'].iloc[medley_idx_rignot].values
smb_uncert_medley_ais_rignot = medley['Surface_Mass_Balance_Uncertainty'].iloc[medley_idx_rignot].values

"""
########################################
# Niwano (Zwally and Rignot, annual, uncertainty)
########################################
niwano = pd.read_csv('/Users/thomas/Documents/github/imbie_partitioning/smb_datasets/Niwano_surface-mass-balance_62578ab94d14f6.50682970/mass-balance-data/NHM-SMAP_v1.0_1979-2021_rate-of-mass-change_IMBIE3_AIS.csv',
                     float_precision = 'round_trip')

# find indices of AIS smb (Zwally region)
niwano_idx_zwally = np.where((niwano['Drainage-Region-Set'] == 'Zwally') & (niwano['Drainage-Region-ID'] == 'AIS'))
# annual time resolution
time_niwano_start = niwano['Start-Date-(decimal-years)'].iloc[niwano_idx_zwally].values
time_niwano_end = niwano['End-Date-(decimal-years)'].iloc[niwano_idx_zwally].values

smb_niwano_ais_zwally = niwano['Rate-of-Mass-Change-(Gt/yr)'].iloc[niwano_idx_zwally].values
smb_uncert_niwano_ais_zwally = niwano['Rate-of-Mass-Change-Uncertainty-(Gt/yr)'].iloc[niwano_idx_zwally].values

# find indices of AIS smb (Rignot region)
niwano_idx_rignot = np.where((niwano['Drainage-Region-Set'] == 'Rignot') & (niwano['Drainage-Region-ID'] == ' AIS'))
smb_niwano_ais_rignot = niwano['Rate-of-Mass-Change-(Gt/yr)'].iloc[niwano_idx_rignot].values
smb_uncert_niwano_ais_rignot = niwano['Rate-of-Mass-Change-Uncertainty-(Gt/yr)'].iloc[niwano_idx_rignot].values
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
smb_vwessem_ais = vwessem['Surface Mass Change'].values
# use racmo 20% uncertainty
smb_uncert_vwessem_ais = 0.2 * smb_vwessem_ais

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
smb_wever_ais_zwally = smb_wever_wais_zwally + smb_wever_eais_zwally + smb_wever_ap_zwally

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
smb_wever_ais_rignot = smb_wever_wais_rignot + smb_wever_eais_rignot + smb_wever_ap_rignot

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
                                   'SMB Zwally':smb_medley_ais_zwally,
                                   'SMB Uncertainty Zwally':smb_uncert_medley_ais_zwally,
                                   'SMB Rignot':smb_medley_ais_rignot,
                                   'SMB Uncertainty Rignot':smb_uncert_medley_ais_rignot})   

datetime_medley_df.set_index('Datetime',
                             inplace = True)

medley_monthly = datetime_medley_df.resample('M').sum()

# create new numpy arrays
time_medley = np.array(medley_monthly.index.year + (medley_monthly.index.dayofyear-1) / 365, dtype = float)
smb_medley_ais_zwally = medley_monthly['SMB Zwally'].values
smb_uncert_medley_ais_zwally = medley_monthly['SMB Uncertainty Zwally'].values
smb_medley_ais_rignot = medley_monthly['SMB Rignot'].values
smb_uncert_medley_ais_rignot = medley_monthly['SMB Uncertainty Rignot'].values

"""
########################################
# Niwano
########################################
datetime_niwano_start = decimal_year_to_datetime(time_niwano_start)
datetime_niwano_end = decimal_year_to_datetime(time_niwano_end)

# resample to monthly
datetime_niwano_df = pd.DataFrame({'Datetime Start':datetime_niwano_start,
                                   'Datetime End':datetime_niwano_end,
                                   'SMB Zwally':smb_niwano_ais_zwally,
                                   'SMB Uncertainty Zwally':smb_uncert_niwano_ais_zwally,
                                   'SMB Rignot':smb_niwano_ais_rignot,
                                   'SMB Uncertainty Rignot':smb_uncert_niwano_ais_rignot})   

datetime_niwano_df['Date'] = datetime_niwano_df.apply(lambda x: pd.date_range(x['Datetime Start'], x['Datetime End'], freq = 'M'), 
                                                      axis=1)

niwano_monthly = (datetime_niwano_df.explode('Date', ignore_index=True).drop(columns=['Datetime Start', 'Datetime End']))

niwano_monthly.set_index('Date',
                         inplace = True)
# create new numpy arrays
time_niwano = np.array(niwano_monthly.index.year + (niwano_monthly.index.dayofyear-1) / 365, dtype = float)
smb_niwano_ais_zwally = niwano_monthly['SMB Zwally'].values / 12
smb_uncert_niwano_ais_zwally = niwano_monthly['SMB Uncertainty Zwally'].values / 12
smb_niwano_ais_rignot = niwano_monthly['SMB Rignot'].values / 12
smb_uncert_niwano_ais_rignot = niwano_monthly['SMB Uncertainty Rignot'].values / 12
"""

########################################
# Wever
########################################
datetime_wever = decimal_year_to_datetime(time_wever)

# resample to monthly
datetime_wever_df = pd.DataFrame({'Datetime':datetime_wever,
                                   'SMB Zwally':smb_wever_ais_zwally,                               
                                   'SMB Rignot':smb_wever_ais_rignot})   

datetime_wever_df.set_index('Datetime',
                             inplace = True)

wever_monthly = datetime_wever_df.resample('M').sum()

# create new numpy arrays
time_wever = np.array(wever_monthly.index.year + (wever_monthly.index.dayofyear-1) / 365, dtype = float)
smb_wever_ais_zwally = wever_monthly['SMB Zwally'].values
smb_wever_ais_rignot = wever_monthly['SMB Rignot'].values


# =============================================================================
# 3. Compare SMB datasets
# =============================================================================
# plot
cmap = plt.cm.get_cmap('Paired')

fig = plt.figure(figsize = (9,4),constrained_layout=True)
gs = plt.GridSpec(1, 1, figure=fig, wspace = 0.1)

ax1 = fig.add_subplot(gs[0])

line_alpha = 0.5

ax1.plot(time_amory, smb_amory_ais_zwally,
         color = cmap(0),
         alpha = line_alpha,
         label = 'Amory Zwally (MAR 3.12.1)',)

ax1.plot(time_hansen, smb_hansen_ais_rignot,
         color = cmap(2),
         alpha = line_alpha,
         label = 'Hansen Rignot (HIRHAM5)')

ax1.plot(time_hansen, smb_hansen_ais_zwally,
         color = cmap(3),
         alpha = line_alpha,
         label = 'Hansen Zwally (HIRHAM5)')

ax1.plot(time_medley, smb_medley_ais_rignot,
         color = cmap(4),
         alpha = line_alpha,
         label = 'Medley Rignot (MERRA-2)')

ax1.plot(time_medley, smb_medley_ais_zwally,
         color = cmap(5),
         alpha = line_alpha,
         label = 'Medley Zwally (MERRA-2)')

ax1.plot(time_vwessem, smb_vwessem_ais,
         color = cmap(6),
         alpha = line_alpha,
         label = 'Van Wessem (RACMO 2.3p2)')

ax1.plot(time_wever, smb_wever_ais_rignot,
         color = cmap(8),
         alpha = line_alpha,
         label = 'Wever Rignot (SNOWPACK)')

ax1.plot(time_wever, smb_wever_ais_zwally,
         color = cmap(9),
         alpha = line_alpha,
         label = 'Wever Zwally (SNOWPACK)')

plt.xlabel('Year')
plt.ylabel('Surface Mass Balance [Gt/month]')

plt.legend(loc = 'center left',
           bbox_to_anchor=(1, 0.5))

plt.savefig('/Users/thomas/Documents/github/imbie_partitioning/figs/smb_ais_datasets.svg', format = 'svg', dpi = 600, bbox_inches='tight')
fig.clf()
plt.close(fig)

# =============================================================================
# 4. Combine SMB datasets
# =============================================================================
# average monthly SMB data
# create dataframes
########################################
# Medley
########################################
medley_monthly = medley_monthly.reset_index()
to_avg_smb_medley_ais_rignot = pd.DataFrame({'Datetime':medley_monthly['Datetime'],
                                             'SMB':medley_monthly['SMB Rignot']})

to_avg_smb_medley_ais_zwally = pd.DataFrame({'Datetime':medley_monthly['Datetime'],
                                             'SMB':medley_monthly['SMB Zwally']})

########################################
# Averaging
########################################
# list of dataframes
df_list = [to_avg_smb_medley_ais_rignot,
           to_avg_smb_medley_ais_zwally]

# set datetime column as the index for each dataframe

for i, df in enumerate(df_list):
    df_list[i] = df.set_index(pd.DatetimeIndex(df['Datetime']))
    
# average dataframes by datetime
smb_average_ais = pd.concat(df_list, axis = 1).mean(axis = 1)