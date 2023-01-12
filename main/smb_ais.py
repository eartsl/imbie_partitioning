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

import matplotlib.pyplot as plt

# =============================================================================
# 1. load AIS SMB datasets
# =============================================================================
########################################
# Amory (Zwally only)
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
# Hansen (Zwally and Rignot)
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
# Medley (Zwally and Rignot)
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


########################################
# Niwano (Zwally and Rignot)
########################################
niwano = pd.read_csv('/Users/thomas/Documents/github/imbie_partitioning/smb_datasets/Niwano_surface-mass-balance_62578ab94d14f6.50682970/mass-balance-data/NHM-SMAP_v1.0_1979-2021_rate-of-mass-change_IMBIE3_AIS.csv',
                     float_precision = 'round_trip')

# find indices of AIS smb (Zwally region)
niwano_idx_zwally = np.where((niwano['Drainage-Region-Set'] == 'Zwally') & (niwano['Drainage-Region-ID'] == 'AIS'))
# annual time resolution
time_niwano = np.floor(niwano['Start-Date-(decimal-years)'].iloc[niwano_idx_zwally]).values
smb_niwano_ais_zwally = niwano['Rate-of-Mass-Change-(Gt/yr)'].iloc[niwano_idx_zwally].values
smb_uncert_niwano_ais_zwally = niwano['Rate-of-Mass-Change-Uncertainty-(Gt/yr)'].iloc[niwano_idx_zwally].values

# find indices of AIS smb (Rignot region)
niwano_idx_rignot = np.where((niwano['Drainage-Region-Set'] == 'Rignot') & (niwano['Drainage-Region-ID'] == 'AIS'))
smb_niwano_ais_rignot = niwano['Rate-of-Mass-Change-(Gt/yr)'].iloc[niwano_idx_rignot].values
smb_uncert_niwano_ais_rignot = niwano['Rate-of-Mass-Change-Uncertainty-(Gt/yr)'].iloc[niwano_idx_rignot].values

########################################
# van Wessem
########################################
ds = xr.open_dataset('/Users/thomas/Documents/github/imbie_partitioning/smb_datasets/van Wessem_surface-mass-balance_6262601bae86e4.66224488/mass-balance-data/ERA5-3H_RACMO2.3p2_ANT27_IMBIE2.nc')
vwessem = pd.DataFrame(np.array([np.arange(1979,2022,1/12), ds.smb[:,-1]]).T,
                       columns = ['Time', 'Surface Mass Change'])
del ds

########################################
# Wever (Zwally and Rignot)
########################################
wever = pd.read_csv('/Users/thomas/Documents/github/imbie_partitioning/smb_datasets/Wever_surface-mass-balance_6268d0772b5c72.42340985/mass-balance-data/imbie.csv',
                    skiprows =  np.arange(107388, 1232037, 1),
                    float_precision = 'round_trip')

# need to sum AP, EAIS, WAIS
# find indices of WAIS smb (Zwally region)
wever_idx_zwally_wais = np.where((wever['Drainage Region Set (Rignot/Zwally)'] == 'Zwally') & (wever['Drainage Region ID'] == 'WAIS'))
# daily time resolution?
time_wever = wever['Date (decimal years)'].iloc[wever_idx_zwally_wais].values
smb_wever_wais_zwally = wever['Relative Mass Change (Gt)'].iloc[wever_idx_zwally_wais].values
smb_uncert_wever_wais_zwally = wever['Relative Mass Change Uncertainty (Gt)'].iloc[wever_idx_zwally_wais].values

# find indices of EAIS smb (Zwally region)
wever_idx_zwally_eais = np.where((wever['Drainage Region Set (Rignot/Zwally)'] == 'Zwally') & (wever['Drainage Region ID'] == 'EAIS'))
smb_wever_eais_zwally = wever['Relative Mass Change (Gt)'].iloc[wever_idx_zwally_eais].values
smb_uncert_wever_eais_zwally = wever['Relative Mass Change Uncertainty (Gt)'].iloc[wever_idx_zwally_eais].values

# find indices of APIS smb (Zwally region)
wever_idx_zwally_ap = np.where((wever['Drainage Region Set (Rignot/Zwally)'] == 'Zwally') & (wever['Drainage Region ID'] == 'AP'))
smb_wever_ap_zwally = wever['Relative Mass Change (Gt)'].iloc[wever_idx_zwally_ap].values
smb_uncert_wever_ap_zwally = wever['Relative Mass Change Uncertainty (Gt)'].iloc[wever_idx_zwally_ap].values

# sum
smb_wever_ais_zwally = smb_wever_wais_zwally + smb_wever_eais_zwally + smb_wever_ap_zwally




