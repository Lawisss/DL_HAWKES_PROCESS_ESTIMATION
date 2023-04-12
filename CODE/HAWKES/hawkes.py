#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Hawkes module

File containing Hawkes Process function (simulation/estimation).

"""

import numpy as np
import Hawkes as hk
from functools import partial
from typing import Tuple, TypedDict

from UTILS.utils import write_csv
import VARIABLES.variables as var

# Simulated Hawkes process 

def hawkes_simulation(params: TypedDict = {"mu": 0.1, "alpha": 0.5, "beta": 10.0}) -> Tuple[hk.simulator, np.ndarray]:
    # Created Hawkes process with a given kernel, baseline and parameters
    hawkes_process = hk.simulator().set_kernel(var.KERNEL).set_baseline(var.BASELINE).set_parameter(params)
    # Simulated Hawkes process in a given time interval
    T = hawkes_process.simulate([var.TIME_ITV_START, var.TIME_HORIZON])
    
    # Plotted the number of events and intensity over time (don't work with many iteration)
    # hawkes_process.plot_N()
    # hawkes_process.plot_l()

    return hawkes_process, T


# Simulated several Hawkes processes

def hawkes_simulations(mu: np.ndarray, alpha: np.ndarray, beta: np.ndarray, filename: str='hawkes_simulations.csv') -> np.ndarray:
    
    # Initialized array to store Hawkes processes (Pre-allocate memory)
    simulated_events_seqs = np.zeros((var.PROCESS_NUM, var.TIME_HORIZON), dtype=np.float32)

    for k in range(var.PROCESS_NUM):
        # Simulated Hawkes processes with the current simulation parameters
        # The results are stored in the k-th row of the simulated_events_seqs array
        _, T = hawkes_simulation(params={"mu": mu[k], "alpha": alpha[k], "beta": beta[k]})
        
        # Converted temporary list T to array and stored results in simulated_events_seqs
        simulated_events_seqs[k,:] = np.asarray(T)[:var.TIME_HORIZON]
    
    # Created dictionaries list representing simulated event sequences
    seqs_list = list(map(partial(lambda _, row: {str(idx): x for idx, x in enumerate(row)}, range(var.TIME_HORIZON)), simulated_events_seqs))

    # Written metrics to a CSV file
    write_csv(seqs_list, filename=filename)

    return simulated_events_seqs


# Estimated Hawkes process

def hawkes_estimation(T: np.ndarray, filename: str = "hawkes_estimation.csv") -> Tuple[np.ndarray, TypedDict, np.ndarray, np.ndarray]:
    
    # Estimated Hawkes process parameters with the given kernel, baseline and parameters
    hawkes_process = hk.estimator().set_kernel(var.KERNEL).set_baseline(var.BASELINE)
    hawkes_process.fit(T, [var.TIME_ITV_START, var.TIME_HORIZON])
    
    # Computed performance metrics for estimated Hawkes process
    metrics = {'Event(s)': len(T),
               'Parameters': {k: round(v, 3) for k, v in hawkes_process.para.items()},
               'Branching Ratio': round(hawkes_process.br, 3),
               'Log-Likelihood': round(hawkes_process.L, 3),
               'AIC': round(hawkes_process.AIC, 3)}
    
    # Written metrics to a CSV file
    write_csv(metrics, filename=filename)

    # Transformed times so that the first observation is at 0 and the last at 1
    [T_transform, interval_transform] = hawkes_process.t_trans() 
    # Predicted the Hawkes process 
    T_pred = hawkes_process.predict(var.END_T, var.NUM_SEQ) 

    # Plotted the empirical survival function of the estimated Hawkes process (don't work with many iteration)
    # hawkes_process.plot_KS()
    # Plotted the predicted number of events over time (don't work with many iteration)
    # hawkes_process.plot_N_pred()

    return T_pred, metrics, T_transform, interval_transform




