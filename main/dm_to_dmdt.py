#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 12:08:13 2023

@author: thomas
"""


def dm_to_dmdt(time, dm, dm_uncert, regression_interval):
    """
    converts dm time series to dmdt by fitting a stepwise linear regression over a user specified time interval
    inputs:
        time = time array in decimal year and monthly temporal resolution [(n,)]
        dm = mass change time series to be converted to mass rates [(n,)]
        dm_uncert = mass change time series uncertainty [(n,)]
        regression_interval = time period over which to fit stepwise linear regression in months (NB must give an integer number of years)    

    outputs:
        dmdt = dmdt time series from dm [(n,)]
        num_points= number of points per epoch used in linear regression [(n,)]

    """

    # import modules

    import numpy as np

    from scipy import stats

    # set window size
    ws = regression_interval / 2

    # get the number of samples within each calendar year
    num_epochs = time.size
    years = np.unique(np.floor(time))
    year_start = years.min()
    year_end = years.max()
    num_years = (year_end - year_start) + 1
    num_samples_year = []

    for i, year in enumerate(years):
        num_samples_year.append(np.sum((time >= year) & (time < year + 1)))
    num_samples_year = np.array(num_samples_year, dtype=float)

    # pad inputs
    time_padded = np.zeros(int(num_epochs + ((regression_interval / 12) * 2)))
    dm_padded = np.zeros(int(num_epochs + ((regression_interval / 12) * 2)))
    dm_uncert_padded = np.zeros(int(num_epochs + ((regression_interval / 12) * 2)))

    time_padded[0:int(regression_interval / 12)] = np.arange(time[0] - (regression_interval / 12), time[0], 1)
    time_padded[int(regression_interval / 12):int((regression_interval / 12) + num_epochs)] = time
    time_padded[int((regression_interval / 12) + num_epochs):] = np.arange(time[-1] + 1,
                                                     time[-1] + (regression_interval / 12) + 1, 1)
    time_padded = time_padded

    dm_padded[0:int(regression_interval / 12)] = dm[0]
    dm_padded[int(regression_interval / 12):int((regression_interval / 12) + num_epochs)] = dm
    dm_padded[int((regression_interval / 12) + num_epochs):] = dm[-1]

    dm_uncert_padded[0:int(regression_interval / 12)] = dm_uncert[0]
    dm_uncert_padded[int(regression_interval / 12):int((regression_interval / 12) + num_epochs)] = dm_uncert
    dm_uncert_padded[int((regression_interval / 12) + num_epochs):] = dm_uncert[-1]

    # define new time vector
    num_months = num_years * 12
    time_step = 1 / 12
    # pad time vector
    time_monthly_padded = np.arange(time[0] - (ws / 12), time[-1] + (ws / 12), time_step).round(4)

    # fit linear regression over regression interval stepped by 1 month
    # initialise arrays
    dmdt = np.full(time_monthly_padded.shape, np.nan)
    dmdt_uncert = np.full(time_monthly_padded.shape, np.nan)
    model_fit_t = [None for _ in time_monthly_padded]
    model_fit_dm = [None for _ in time_monthly_padded]
    model_fit_num_points = [0 for _ in time_monthly_padded]

    # linear model function
    def lscov(
        a: np.ndarray, b: np.ndarray, v: np.ndarray = None, dx: bool = False
    ) -> np.ndarray:
        """
        This is a python implementation of the matlab lscov function. This has been written based upon the matlab source
        code for lscov.m, which can be found here: http://opg1.ucsd.edu/~sio221/SIO_221A_2009/SIO_221_Data/Matlab5/Toolbox/matlab/matfun/lscov.m
        """

        m, n = a.shape
        if m < n:
            raise Exception(
                f"problem must be over-determined so that M > N. ({m}, {n})")
        if v is None:
            v = np.eye(m)

        if v.shape != (m, m):
            raise Exception("v must be a {0}-by-{0} matrix".format(m))

        qnull, r = qr(a, mode="complete")
        q = qnull[:, :n]
        r = r[:n, :n]

        qrem = qnull[:, n:]
        g = qrem.T.dot(v).dot(qrem)
        f = q.T.dot(v).dot(qrem)

        c = q.T.dot(b)
        d = qrem.T.dot(b)

        x = solve(r, (c - f.dot(solve(g, d))))

        # This was not required for merge_dM, and so has been removed as it has problems.
        if dx:
            u = cholesky(v).T
            z = solve(u, b)
            w = solve(u, a)
            mse = (z.T.dot(z) - x.T.dot(w.T.dot(z))) / (m - n)
            q, r = qr(w)
            ri = solve(r, np.eye(n)).T
            dx = np.sqrt(np.sum(ri * ri, axis=0) * mse).T

            return x, dx
        return x

    for i, t in enumerate(time_monthly_padded):

        if (t >= time_monthly_padded[0] + (ws / 12) and (t < time_monthly_padded[-1] - (ws / 12))):

            window_start = t - (ws / 12)
            window_end = t + (ws / 12)
            in_window = np.where((time_padded >= window_start)
                                    & (time_padded < window_end))

            window_t = time_padded[in_window]
            window_dm = dm_padded[in_window]
            window_dm_uncert = dm_uncert_padded[in_window]
        

            # fit linear trend to get dm/dt with error weighting
            # check for nans
            is_data = np.isfinite(window_dm)
            if is_data.size > 1:
                # prepare for fit
                lsq_fit = np.vstack([np.ones_like(window_t), window_t]).T
                # error-weighted LSQ fitting
                w = np.diag(1.0 / np.square(window_dm_uncert))
                lsq_coef, lsq_se = lscov(lsq_fit, window_dm, w, dx=True)

                dmdt[i] = lsq_coef[1]

                # get RMS of input errors in window
                window_dm_uncert_rms = np.sqrt(np.nanmean(window_dm_uncert ** 2))
                dmdt_uncert[i] = np.sqrt(lsq_se[1] ** 2 + window_dm_uncert_rms ** 2)

                # store data used in fit
                model_fit_t[i] = np.r_[window_start:window_end:0.2]
                model_fit_dm[i] = lsq_coef[0] + lsq_coef[1] * model_fit_t[i]
                model_fit_num_points[i] = is_data.sum()

            else:
                continue

        else:
            continue

    # truncate padded values
    to_keep = np.where((time_monthly_padded >= time[0]) & (time_monthly_padded < time[-1]))

    time_out = time_monthly_padded[to_keep]
    dmdt_out = dmdt[to_keep]
    dmdt_uncert_out = dmdt_uncert[to_keep]
    out_array = np.vstack((time_out,dmdt_out,dmdt_uncert_out)).T

    return