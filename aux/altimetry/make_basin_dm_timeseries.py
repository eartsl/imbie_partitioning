#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Routine to take pixel level filled but not cross-calibrated dh data, and create basin dm timeseries.
This follows the IMBIE 2020 basin dm timeseries code as far as possible, but uses Tom Slater's
new part-static dynamic ice mask (ie only Pine Island and Thwaites evolve over time, while Totten, Getz and the AP areas
remain static). The error in the pole hole fill that was present in the 2020 code is fixed here. There are
increased dh and dh/dt sigma limits compared to the 2020 interpolation. Finally, there are 3 extra basins that
collectively contain Totten, from the MEaSUREs basins.

Author: Lin Gilbert (UCL/MSSL/CPOM)
Date: 05/04/2022
Copyright: UCL/MSSL/CPOM. Not to be used outside UCL/MSSL/CPOM without permission of author
"""
# run in cpom conda environment
import scipy.io as sio
from netCDF4 import Dataset
import numpy as np
import copy
from scipy import stats
import statsmodels.api as sm

# Set up main data directory, which contains inputs and outputs for this routine

# note must be connected to /Volumes/a236
maindir = '/Volumes/a236/cpnet/altimetry/landice/imbie_2022/'

# Read in dh data

dh_in = np.load(maindir+'pixel_dh_timeseries_filled.npz')
dh = dh_in['dh']
sd = dh_in['sd']
flg = dh_in['flg']
t = dh_in['t']
nt = dh_in['nt']
m = dh_in['m']

t = t+1991.0   # Work in J2000

r = np.where(flg > 2)  # Remove fills to mimic IMBIE 2020
dh[r] = np.nan
sd[r] = np.nan
flg[r] = 0

# Get MEaSUREs basins, pole hole mask and dynamic ice mask.
ph = sio.readsav(maindir+'/pole_hole_mask_5km.sav')
ph_mask = ph['ph_mask']

datadi = Dataset(
    '/Volumes/a236/cpdata/RESOURCES/surface_discrimination_masks/antarctica/dynamic_ice_mask/ais_density_mask_1992_2021_part_static.nc')
di_mask = datadi['year'][:].data


measures_mask = np.load(
    '/Users/thomas/Documents/github/imbie_partitioning/aux/altimetry/output/measures_mask.npz', allow_pickle=True)
basins = measures_mask['measures_mask']
basin_names = measures_mask['basin_names']

datafd = Dataset(
    '/Volumes/a236/cpdata/MODELS/RACMO/firn_density/Rho0_2011/ANT3K27_Rho0_5km.nc')
fd_mask = datafd['Rho0'][:].data

# Set up some known pole hole info

basins_in_ph = np.array([8, 9, 16, 17])
num_basins_in_ph = basins_in_ph.size

# Convert dh to dm - if it's an ice area, use fixed ice density, otherwise use the firn density mask. Flag ice/snow.
# Also convert the stddev. For now, leave in kg.

ice_dens = 917.0   # kg/m3
cell_area = 5e3*5e3   # Area of 5km cell, in m2

dm = np.full_like(dh, np.nan)
dm_sd = np.full_like(dh, np.nan)
icesnow_flag = np.zeros_like(dh)

for i in range(nt):

    this_dh = copy.deepcopy(dh[i, :, :])
    this_sd = copy.deepcopy(sd[i, :, :])

    # Make a binary snow/ice mask for each epoch
    this_di = np.zeros_like(di_mask)
    r = np.where(di_mask <= t[i])
    this_di[r] = 1

    this_icesnow_flag = np.zeros_like(this_di)

    r = np.where(np.isfinite(this_dh) & (this_di == 1))

    this_icesnow_flag[r] = 1
    this_dh[r] = this_dh[r]*cell_area*ice_dens
    this_sd[r] = this_sd[r]*cell_area*ice_dens

    r = np.where(np.isfinite(this_dh) & (this_di == 0))

    this_dh[r] = this_dh[r]*cell_area*fd_mask[r]
    this_sd[r] = this_sd[r]*cell_area*fd_mask[r]

    dm[i, :, :] = copy.deepcopy(this_dh)
    dm_sd[i, :, :] = copy.deepcopy(this_sd)
    icesnow_flag[i, :, :] = copy.deepcopy(this_icesnow_flag)

# Make the basin timeseries for 2022. To mimic 2020 as far as possible, only use datapoints flagged 1 or 2
#############################
#############################
#############################

num_basins = len(basin_names)

dm_ice = np.full([num_basins, nt], np.nan)
dm_snow = np.full([num_basins, nt], np.nan)
dm_both = np.full([num_basins, nt], np.nan)

num_ice_obs = np.zeros([num_basins, nt])   # Obs = observed or interpolated
num_snow_obs = np.zeros([num_basins, nt])
num_both_obs = np.zeros([num_basins, nt])
num_ice_all = np.zeros([num_basins, nt])   # All = obs + gaps
num_snow_all = np.zeros([num_basins, nt])
num_both_all = np.zeros([num_basins, nt])

snow_ph_dm = np.full(nt, np.nan)

for b in range(num_basins):

    print('Collecting data from basin ', basin_names[b])

    for i in range(nt):

        this_dm = copy.deepcopy(dm[i, :, :])
        this_flg = copy.deepcopy(flg[i, :, :])

        # Slot in pole hole values - CS2 has a smaller pole hole. This is now in all basins except 18

        if m[i] < 3:
            r = np.where((basins > 0) & (basins != 18) & (
                (ph_mask == 1) | (ph_mask == 2)) & np.isnan(di_mask))
        else:
            r = np.where((basins > 0) & (basins != 18) &
                         (ph_mask == 2) & np.isnan(di_mask))

        q = np.where((basins > 0) & (basins != 18) & (ph_mask == 3)
                     & np.isnan(di_mask) & np.isfinite(this_dm))

        if len(q[0]) > 0:
            this_dm[r] = np.nanmean(this_dm[q])
            snow_ph_dm[i] = np.nanmean(this_dm[q])

        # Collect ice and snow areas separately, using nearest-in-time density mask

        this_di = np.zeros_like(di_mask)   # Make a binary mask for each epoch
        r = np.where(di_mask <= t[i])
        this_di[r] = 1

        r = np.where((basins == b+1) & np.isfinite(this_dm) & (this_di == 1))
        q = np.where((basins == b+1) & (this_di == 1))

        dm_ice[b, i] = np.sum(this_dm[r])
        num_ice_obs[b, i] = len(r[0])
        num_ice_all[b, i] = len(q[0])

        # Do the same for snow
        r = np.where((basins == b+1) & np.isfinite(this_dm) & (this_di == 0))
        q = np.where((basins == b+1) & (this_di == 0))

        dm_snow[b, i] = np.sum(this_dm[r])
        num_snow_obs[b, i] = len(r[0])
        num_snow_all[b, i] = len(q[0])

        # Add snow and ice to get 'both'. If only one exists, repeat that one.

        if np.isfinite(dm_ice[b, i]) and np.isfinite(dm_snow[b, i]):
            dm_both[b, i] = dm_ice[b, i]+dm_snow[b, i]
            num_both_obs[b, i] = num_ice_obs[b, i]+num_snow_obs[b, i]
            num_both_all[b, i] = num_ice_all[b, i]+num_snow_all[b, i]

        if np.isfinite(dm_ice[b, i]) and np.isnan(dm_snow[b, i]):
            dm_both[b, i] = dm_ice[b, i]
            num_both_obs[b, i] = num_ice_obs[b, i]
            num_both_all[b, i] = num_ice_all[b, i]

        if np.isnan(dm_ice[b, i]) and np.isfinite(dm_snow[b, i]):
            dm_both[b, i] = dm_snow[b, i]
            num_both_obs[b, i] = num_snow_obs[b, i]
            num_both_all[b, i] = num_snow_all[b, i]

# Scale up ie take the mean of ice, or snow, or ice+snow, and multiply
# by corresponding number of cells in basin at that date. The pole hole
# has got the equivalent of a single cell value so far, so multiply up
# by the number of cells in the pole hole (no ice evolution in pole hole).

dm_ice = (dm_ice/num_ice_obs)*num_ice_all
dm_snow = (dm_snow/num_snow_obs)*num_snow_all
dm_both = (dm_both/num_both_obs)*num_both_all

# Put in the cross-calibration

xcal_dm_ice = np.full_like(dm_ice, np.nan)
xcal_dm_snow = np.full_like(dm_snow, np.nan)
xcal_dm_both = np.full_like(dm_both, np.nan)

# Make independent variables, same for all timeseries - only flag last 3 missions
iv = np.zeros((nt, 6))
iv[:, 0] = t
iv[:, 1] = t**2
iv[:, 2] = t**3
for mission in range(1, 4):
    ok = np.where(m == mission)
    iv[ok, mission+2] = 1
iv = sm.add_constant(iv, has_constant='add')

weight = np.ones_like(t)   # Equal weighting

xcal_coeffs_ice = np.full([num_basins, 7], np.nan)    # Save coeffs
xcal_coeffs_snow = np.full([num_basins, 7], np.nan)
xcal_coeffs_both = np.full([num_basins, 7], np.nan)

r1 = np.where(m == 1)  # Get mission array locations, for applying biases
r2 = np.where(m == 2)
r3 = np.where(m == 3)

for b in range(num_basins):

    print('Cross-calibrating basin ', basin_names[b])

    # Start with ice, if it exists

    if np.sum(np.isfinite(dm_ice[b, :])) > 0:

        this_iv = copy.deepcopy(iv)
        this_weight = copy.deepcopy(weight)
        this_dm = copy.deepcopy(dm_ice[b, :])

        if b == 10:                       # Pine Island starts with no ice, so can't use full timeseries
            r = np.where(np.isfinite(this_dm))
            this_iv = this_iv[r, :]
            # Me neither - applying r adds a dimension of 1 to iv, but not to weight or dm
            this_iv = this_iv[0, :, :]
            this_weight = this_weight[r]
            this_dm = this_dm[r]

        try:
            res = sm.WLS(this_dm, this_iv, weights=this_weight,
                         missing='drop').fit().summary()
        except:
            print('ICE CROSS_CALIBRATION FAILED, BASIN', b)
        else:
            res = sm.WLS(this_dm, this_iv, weights=this_weight,
                         missing='drop').fit()
            xcal_coeffs_ice[b, :] = res.params

        xcal_dm_ice[b, :] = copy.deepcopy(dm_ice[b, :])
        xcal_dm_ice[b, r1] = xcal_dm_ice[b, r1]-xcal_coeffs_ice[b, 4]
        xcal_dm_ice[b, r2] = xcal_dm_ice[b, r2]-xcal_coeffs_ice[b, 5]
        xcal_dm_ice[b, r3] = xcal_dm_ice[b, r3]-xcal_coeffs_ice[b, 6]

    # Then snow, which always exists

    this_iv = copy.deepcopy(iv)
    this_weight = copy.deepcopy(weight)
    this_dm = copy.deepcopy(dm_snow[b, :])

    try:
        res = sm.WLS(this_dm, this_iv, weights=this_weight,
                     missing='drop').fit().summary()
    except:
        print('SNOW CROSS_CALIBRATION FAILED, BASIN', b)
    else:
        res = sm.WLS(this_dm, this_iv, weights=this_weight,
                     missing='drop').fit()
        xcal_coeffs_snow[b, :] = res.params

    xcal_dm_snow[b, :] = copy.deepcopy(dm_snow[b, :])
    xcal_dm_snow[b, r1] = xcal_dm_snow[b, r1]-xcal_coeffs_snow[b, 4]
    xcal_dm_snow[b, r2] = xcal_dm_snow[b, r2]-xcal_coeffs_snow[b, 5]
    xcal_dm_snow[b, r3] = xcal_dm_snow[b, r3]-xcal_coeffs_snow[b, 6]

    # Then both

    this_iv = copy.deepcopy(iv)
    this_weight = copy.deepcopy(weight)
    this_dm = copy.deepcopy(dm_both[b, :])

    try:
        res = sm.WLS(this_dm, this_iv, weights=this_weight,
                     missing='drop').fit().summary()
    except:
        print('BOTH CROSS_CALIBRATION FAILED, BASIN', b)
    else:
        res = sm.WLS(this_dm, this_iv, weights=this_weight,
                     missing='drop').fit()
        xcal_coeffs_both[b, :] = res.params

    xcal_dm_both[b, :] = copy.deepcopy(dm_both[b, :])
    xcal_dm_both[b, r1] = xcal_dm_both[b, r1]-xcal_coeffs_both[b, 4]
    xcal_dm_both[b, r2] = xcal_dm_both[b, r2]-xcal_coeffs_both[b, 5]
    xcal_dm_both[b, r3] = xcal_dm_both[b, r3]-xcal_coeffs_both[b, 6]

# And we have to make a final version with epochs where missions overlap averaged. For the number of observed/interpolated
# cells this is difficult, as we've lost the link to which cells were seen by both missions and which were only seen by one.
# Take the larger value of the two, as that number has definitely been used.

mission_name = ['e1', 'e2', 'ev', 'cs2']

final_t = np.unique(t)
final_nt = final_t.size

# Couldn't get a numpy array of dtype=object (the only way to do strings) to work here, so use a list
final_m = [None]*final_nt

final_xcal_dm_ice = np.full([num_basins, final_nt], np.nan)
final_xcal_dm_snow = np.full([num_basins, final_nt], np.nan)
final_xcal_dm_both = np.full([num_basins, final_nt], np.nan)

final_num_ice_obs = np.zeros([num_basins, final_nt])  # Observed + interpolated
final_num_snow_obs = np.zeros([num_basins, final_nt])
final_num_both_obs = np.zeros([num_basins, final_nt])

final_num_ice_all = np.zeros([num_basins, final_nt])
final_num_snow_all = np.zeros([num_basins, final_nt])
final_num_both_all = np.zeros([num_basins, final_nt])

for i in range(final_nt):

    r = np.where(t == final_t[i])

    if len(r[0]) == 1:

        final_m[i] = mission_name[int(m[r[0]])]

        final_xcal_dm_ice[:, i] = xcal_dm_ice[:, r[0][0]]
        final_xcal_dm_snow[:, i] = xcal_dm_snow[:, r[0][0]]
        final_xcal_dm_both[:, i] = xcal_dm_both[:, r[0][0]]

        final_num_ice_obs[:, i] = num_ice_obs[:, r[0][0]]
        final_num_snow_obs[:, i] = num_snow_obs[:, r[0][0]]
        final_num_both_obs[:, i] = num_both_obs[:, r[0][0]]

        final_num_ice_all[:, i] = num_ice_all[:, r[0][0]]
        final_num_snow_all[:, i] = num_snow_all[:, r[0][0]]
        final_num_both_all[:, i] = num_both_all[:, r[0][0]]

    else:

        m_str = mission_name[int(m[r[0][0]])]
        for j in range(1, r[0].size):
            # Gordon Bennett! But it works.
            m_str = m_str+' '+mission_name[int(m[r[0][j]])]
        final_m[i] = m_str

        final_xcal_dm_ice[:, i] = np.mean(xcal_dm_ice[:, r[0]], axis=1)
        final_xcal_dm_snow[:, i] = np.mean(xcal_dm_snow[:, r[0]], axis=1)
        final_xcal_dm_both[:, i] = np.mean(xcal_dm_both[:, r[0]], axis=1)

        final_num_ice_obs[:, i] = np.nanmax(
            num_ice_obs[:, r[0]], axis=1)   # Take largest value
        final_num_snow_obs[:, i] = np.nanmax(num_snow_obs[:, r[0]], axis=1)
        final_num_both_obs[:, i] = np.nanmax(num_both_obs[:, r[0]], axis=1)

        # Number of cells in this epoch is same for all missions
        final_num_ice_all[:, i] = num_ice_all[:, r[0][0]]
        final_num_snow_all[:, i] = num_snow_all[:, r[0][0]]
        final_num_both_all[:, i] = num_both_all[:, r[0][0]]

final_xcal_dm_both = final_xcal_dm_both/1e12   # Convert to Gt
final_xcal_dm_ice = final_xcal_dm_ice/1e12
final_xcal_dm_snow = final_xcal_dm_snow/1e12

# Save

np.savez('/Users/thomas/Documents/github/imbie_partitioning/aux/altimetry/output/altimetry_dm_measures_basins.npz', t=final_t, dm_snowice=final_xcal_dm_both, dm_snow=final_xcal_dm_snow,
         dm_ice=final_xcal_dm_ice, num_snowice_obs=final_num_both_obs, num_snow_obs=final_num_snow_obs,
         num_ice_obs=final_num_ice_obs, num_snowice_all=final_num_both_all, num_snow_all=final_num_snow_all,
         num_ice_all=final_num_ice_all)
