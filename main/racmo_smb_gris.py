#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 10:22:33 2023

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

import xarray as xr

import numpy as np

import pandas as pd

from datetime import datetime, timedelta

from timeit import default_timer as timer


# =============================================================================
# 1. load racmo
# =============================================================================
# load ice sheet mask
ds = xr.open_dataset('/Volumes/eartsl/smb_models/racmo/greenland/jan_2023/Icemask_Topo_Iceclasses_lon_lat_average_1km_GrIS.nc')
x = ds.x.values
y = ds.y.values
Promicemask = ds.Promicemask.values
del ds

icesheetmask = Promicemask == 3

# get racmo smb filenames
path = '/Volumes/eartsl/smb_models/racmo/greenland/jan_2023/Monthly-1km'
filelist = [os.path.join(path, file) for file in os.listdir(path) if file.endswith('.nc')]

# =============================================================================
# 2. accumulate gridded racmo data to create time series for ice sheet
# =============================================================================
start = timer()

# intialise arrays

time = []
smb = []

for i, file in enumerate(filelist):
    print(file)
    
    if i < 32:
    
        ds = xr.open_dataset(file,
                             decode_times = False)
        
        units, reference_date = ds.time.attrs['units'].split('since ')
        reference_datetime = datetime.strptime(reference_date, '%Y-%m-%d %H:%M:%S')
        t0 = reference_datetime.year + (reference_datetime.day - 1) / 365
        months_since_t0 = ds.time.values
        
        smb_rec = ds.SMB_rec.values
        
    else:
     
        ds = xr.open_dataset(file,
                             decode_times = False)
        
        units, reference_date = ds.time.attrs['units'].split('since ')
        reference_datetime = datetime.strptime(reference_date, '%Y-%m-%d %H:%M:%S')
        t0 = reference_datetime.year + (reference_datetime.day - 1) / 365
        months_since_t0 = ds.time.values / 30.44
        
        smb_rec = ds.smb_rec.values
        
    # accumulate SMB
    for ii, t in enumerate(months_since_t0):
        
        time.append(t0 + (t / 12))
        
        smb_tmp = smb_rec[ii,:,:]
        # convert from mmWE to Gt
        smb_tmp = smb_tmp * 1e3 * 1e3 * 1e-12
        smb.append(np.ma.masked_array(smb_tmp, ~icesheetmask).sum())
        
    del ds
    
end = timer()
print(timedelta(seconds=end-start))

# =============================================================================
# 3. Save as csv
# =============================================================================
        
df_out = pd.DataFrame({'Date (Decimal Years)':np.array(time, dtype = float),
                       'Surface Mass Change (Gt)':np.array(smb, dtype = float)})

df_out.to_csv('/Users/thomas/Documents/github/imbie_partitioning/smb_datasets/Noel_aggregated_smb/ERA5-3H_RACMO2.3p2_GrIS1_IMBIE2.csv',
              index = False)