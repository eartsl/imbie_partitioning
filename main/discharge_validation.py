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
import pandas as pd
import numpy as np
import os
import sys
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
dmdt_smb_uncert_imbie_ais = imbie_ais['Surface mass balance anomaly uncertainty (Gt/yr)']

# load aggregated time series for indivdual techniques
techniques_ais = pd.read_csv('/Users/thomas/Documents/github/imbie_partitioning/aux/imbie_datasets/individual_techniques_aggregated/ais.csv',
                            float_precision='round_trip',
                            header=None)

# Altimetry
time_ra_ais = techniques_ais[2].iloc[np.where(techniques_ais[0] == 'RA')].values
dmdt_ra_ais = techniques_ais[3].iloc[np.where(techniques_ais[0] == 'RA')].values
dmdt_uncert_ra_ais = techniques_ais[4].iloc[np.where(techniques_ais[0] == 'RA')].values

# GMB
time_gmb_ais = techniques_ais[2].iloc[np.where(techniques_ais[0] == 'GMB')].values
dmdt_gmb_ais = techniques_ais[3].iloc[np.where(techniques_ais[0] == 'GMB')].values
dmdt_uncert_gmb_ais = techniques_ais[4].iloc[np.where(techniques_ais[0] == 'GMB')].values

# compute discharge for Altimetry and GMB
# Altimetry
# interpolate SMB
dmdt_smb_ra_ais = np.interp(time_ra_ais,
                        time_imbie_ais,
                        dmdt_smb_imbie_ais)

dmdt_dyn_ra_ais = dmdt_ra_ais - dmdt_smb_ra_ais

# GMB
# interpolate SMB
dmdt_smb_gmb_ais = np.interp(time_gmb_ais,
                        time_imbie_ais,
                        dmdt_smb_imbie_ais)

dmdt_dyn_gmb_ais = dmdt_gmb_ais - dmdt_smb_gmb_ais

# load IOM discharge
