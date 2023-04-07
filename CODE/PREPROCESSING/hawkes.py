#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Hawkes module

File containing Hawkes Process function (simulation/estimation).

"""

import numpy as np
import pandas as pd
import Hawkes as hk

from UTILS.utils import write_csv
import VARIABLES.variables as var

# Simulated Hawkes process 

def hawkes_simulation(params={"mu": 0.1, "alpha": 0.5, "beta": 10.0}):
    # Created a Hawkes process with the given kernel, baseline and parameters
    hawkes_process = hk.simulator().set_kernel(var.KERNEL).set_baseline(var.BASELINE).set_parameter(params)
    # Simulated a Hawkes process in the given time interval
    T = hawkes_process.simulate([var.TIME_ITV_START, var.TIME_HORIZON])
    
    # Plotted the number of events and intensity over time (don't work with many iteration)
    # hawkes_process.plot_N()
    # hawkes_process.plot_l()

    return hawkes_process, T


# Simulated several Hawkes processes

def hawkes_simulations(mu, alpha, beta, filename='hawkes_simulations.csv'):
    # Initialize a filled with zeros array to store Hawkes processes (Pre-allocate memory)
    simulated_events_seqs = np.zeros((var.PROCESS_NUM, var.TIME_HORIZON), dtype=np.float64)

    for k in range(var.PROCESS_NUM):
        # Simulate a Hawkes processes with the current simulation parameters
        # The results are stored in the k-th row of the simulated_events_seqs array
        _, T = hawkes_simulation(params={"mu": mu[k], "alpha": alpha[k], "beta": beta[k]})
        
        # Convert temporary list T to an array and store the results in simulated_events_seqs
        simulated_events_seqs[k,:] = np.asarray(T)[:var.TIME_HORIZON]

    # Created a DataFrame, name the columns, and generate csv file
    df = pd.DataFrame(np.row_stack(simulated_events_seqs))
    df.to_csv(f"{var.FILEPATH}{filename}", index=False)

    return simulated_events_seqs


# Estimated Hawkes process

def hawkes_estimation(T, filename="hawkes_estimation.csv"):
    
    # Estimated Hawkes process parameters with given kernel and baseline
    hawkes_process = hk.estimator().set_kernel(var.KERNEL).set_baseline(var.BASELINE)
    hawkes_process.fit(T, [var.TIME_ITV_START, var.TIME_HORIZON])
    
    # Computed performance metrics for the estimated Hawkes process
    metrics = {'Event(s)': len(T),
               'Parameters': {k: round(v, 3) for k, v in hawkes_process.para.items()},
               'Branching Ratio': round(hawkes_process.br, 3),
               'Log-Likelihood': round(hawkes_process.L, 3),
               'AIC': round(hawkes_process.AIC, 3)}
    
    # Written metrics to a CSV file
    write_csv(metrics, f"{var.FILEPATH}{filename}")

    # Transformed times so that the first observation is at 0 and the last at 1
    [T_transform, interval_transform] = hawkes_process.t_trans() 
    # Predicted the Hawkes process 
    T_pred = hawkes_process.predict(var.END_T, var.NUM_SEQ) 

    # Plotted the empirical survival function of the estimated Hawkes process (don't work with many iteration)
    # hawkes_process.plot_KS()
    # Plotted the predicted number of events over time (don't work with many iteration)
    # hawkes_process.plot_N_pred()

    return T_pred, metrics, T_transform, interval_transform