#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 12:08:13 2023

@author: thomas
"""

def dm_to_dmdt(time, dm, regression_interval):
    
    """
    converts dm time series to dmdt by fitting a stepwise linear regression over a user specified time interval
    inputs:
        time = time array in decimal year and monthly temporal resolution [(n,)]
        dm = mass change time series to be converted to mass rates [(n,)]
        regression_interval = time period over which to fit stepwise linear regression (default 36 months)        

    outputs:
        dmdt = dmdt time series from dm [(n,)]
        num_points= number of points per epoch used in linear regression [(n,)]

    """
    
    # import modules
    
    import numpy as np
    
    from scipy import stats
        
    # set window size
    ws = regression_interval / 2
    
    # initialise arrays
    dmdt = []
    # number of data points used in fit
    num_points = []
    
    # function to find index of nearest input value in a given vector
    def dsearchn(x, v):
        return int(np.where(np.abs(x - v) == np.abs(x - v).min())[0])
    
    # fit linear regression over regression interval stepped by 1 month
    for i, t in enumerate(time):
        # don't run if can't fit over full regression interval period
        if (t > time[0] + (ws / 12)) and (t < time[-1] - (ws / 12)):
            
            # find window (centred on t[i]) to fit linear regression
            t1 = dsearchn(time, t - (ws / 12))
            t2 = dsearchn(time, t + (ws / 12))
            
            reg = stats.linregress(time[t1:t2],
                                   dm[t1:t2])
            
            dmdt.append(reg.slope)
            num_points.append(np.isfinite(dm[t1:t2]).sum())
            
        else: 
            dmdt.append(np.nan)
            num_points.append(0)
            
    dmdt = np.array(dmdt, dtype = float)
    num_points = np.array(num_points, dtype = float)
    
    # pad truncated bits with first and last values
    dmdt[np.where((num_points == 0)) and (time <= (time[0] + (ws / 12)))] = dmdt[np.isfinite(dmdt)][0]
    dmdt[np.where((num_points == 0)) and (time >= (time[-1] - (ws / 12)))] = dmdt[np.isfinite(dmdt)][-1]
            
    return dmdt, num_points