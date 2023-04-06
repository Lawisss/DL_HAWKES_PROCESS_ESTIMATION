#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Hawkes module

File containing Hawkes Process function (simulation/estimation).

"""

import numpy as np
import pandas as pd
import Hawkes as hk

from UTILS.utils import write_csv

# Simulated Hawkes process and plots the number of events / intensity function over time

def hawkes_simulation(kernel='exp', baseline='const', params={"mu": 0.1, "alpha": 0.5, "beta": 10.0}, interval=[0,100]):
    # Created a Hawkes process with the given kernel, baseline and parameters
    hawkes_process = hk.simulator().set_kernel(kernel).set_baseline(baseline).set_parameter(params)
    # Simulated a Hawkes process in the given time interval
    T = hawkes_process.simulate(interval)
    
    # Plotted the number of events and intensity over time (don't work with many iteration)
    # hawkes_process.plot_N()
    # hawkes_process.plot_l()

    return hawkes_process, T


# Estimated Hawkes process and plots the number of events / intensity function over time

def hawkes_estimation(T, kernel='exp', baseline='const', interval=[0,100], end_T=200, num_seq=100, filepath="C:/Users/Nicolas Girard/Documents/VAE_HAWKES_PROCESS_ESTIMATION/CODE/RESULTS/hawkes_estimation.csv"):
    
    # Estimated Hawkes process parameters with given kernel and baseline
    hawkes_process = hk.estimator().set_kernel(kernel).set_baseline(baseline)
    hawkes_process.fit(T, interval)
    
    # Computed performance metrics for the estimated Hawkes process
    metrics = {'Event(s)': len(T),
               'Parameters': {k: round(v, 3) for k, v in hawkes_process.para.items()},
               'Branching Ratio': round(hawkes_process.br, 3),
               'Log-Likelihood': round(hawkes_process.L, 3),
               'AIC': round(hawkes_process.AIC, 3)}
    
    # Written metrics to a CSV file
    write_csv(metrics, filepath)

    # Transformed times so that the first observation is at 0 and the last at 1
    [T_transform, interval_transform] = hawkes_process.t_trans() 
    # Predicted the Hawkes process 
    T_pred = hawkes_process.predict(end_T, num_seq) 

    # Plotted the empirical survival function of the estimated Hawkes process (don't work with many iteration)
    # hawkes_process.plot_KS()
    # Plotted the predicted number of events over time (don't work with many iteration)
    # hawkes_process.plot_N_pred()

    return T_pred, metrics, T_transform, interval_transform