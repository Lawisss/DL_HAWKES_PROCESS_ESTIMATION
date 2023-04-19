#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Hawkes module

File containing Hawkes process function (simulation/estimation)

"""

from typing import Tuple, TypedDict

import numpy as np
import Hawkes as hk

from UTILS.utils import write_parquet
import VARIABLES.hawkes_var as hwk

# Simulated Hawkes process 

def hawkes_simulation(params: TypedDict = {"mu": 0.1, "alpha": 0.5, "beta": 10.0}) -> Tuple[hk.simulator, np.ndarray]:
    
    """
    Simulated Hawkes process with given parameters

    Args:
        params (TypedDict, optional): Parameters of Hawkes process. Default is {"mu": 0.1, "alpha": 0.5, "beta": 10.0}

    Returns:
        Tuple[hk.simulator, np.ndarray]: Hawkes process simulator and the simulated times
    """

    # Created Hawkes process with a given kernel, baseline and parameters
    hawkes_process = hk.simulator().set_kernel(hwk.KERNEL).set_baseline(hwk.BASELINE).set_parameter(params)
    # Simulated Hawkes process in a given time interval
    t = hawkes_process.simulate([hwk.TIME_ITV_START, hwk.TIME_HORIZON])
    
    # Plotted the number of events and intensity over time (don't work with many iteration)
    # hawkes_process.plot_N()
    # hawkes_process.plot_l()

    return hawkes_process, t


# Simulated several Hawkes processes

def hawkes_simulations(alpha: np.ndarray, beta: np.ndarray, mu: np.ndarray, filename: str='hawkes_simulations.parquet') -> np.ndarray:
    
    """
    Simulated several Hawkes processes using parameters, and saved results to Parquet file 

    Args:
        alpha (np.ndarray): Excitation matrix of each Hawkes process
        beta (np.ndarray): Decay matrix of each Hawkes process
        mu (np.ndarray): Base intensity of each Hawkes process
        filename (str, optional): Parquet filename to save results. Defaults to 'hawkes_simulations.parquet'

    Returns:
        np.ndarray: Simulated event sequences of each Hawkes process
    """

    # Initialized array to store Hawkes processes (Pre-allocate memory)
    simulated_events_seqs = np.zeros((hwk.PROCESS_NUM, hwk.TIME_HORIZON), dtype=np.float32)

    for k in range(hwk.PROCESS_NUM):
        # Simulated Hawkes processes with the current simulation parameters
        # The results are stored in the k-th row of the simulated_events_seqs array
        _, t = hawkes_simulation(params={"mu": mu[k], "alpha": alpha[k], "beta": beta[k]})
        
        # Length clipping to not exceed time horizon
        seq_len = np.minimum(np.size(t), hwk.TIME_HORIZON)
        simulated_events_seqs[k,:seq_len] = t[:seq_len]
    
    # Written parameters to Parquet file
    write_parquet(simulated_events_seqs, columns=np.arange(hwk.TIME_HORIZON, dtype=np.int32).astype(str), filename=filename)

    # Created dictionaries list representing simulated event sequences
    # seqs_list = list(map(partial(lambda _, row: {str(idx): x for idx, x in enumerate(row)}, range(hwk.TIME_HORIZON)), simulated_events_seqs))

    # Written metrics to a CSV file
    # write_csv(seqs_list, filename=filename)

    return simulated_events_seqs


# Estimated Hawkes process

def hawkes_estimation(t: np.ndarray, filename: str = "hawkes_estimation.parquet") -> Tuple[np.ndarray, TypedDict, np.ndarray, np.ndarray]:
    
    """
    Estimated Hawkes process from event times, returned predicted process and performance metrics

    Args:
        t (np.ndarray): Event times
        filename (str, optional): Parquet filename for performance metrics. Defaults to "hawkes_estimation.parquet"

    Returns:
        Tuple[np.ndarray, TypedDict, np.ndarray, np.ndarray]: A tuple containing the following items:
            - t_pred (np.ndarray): Predicted event times for estimated Hawkes process
            - metrics (TypedDict): Performance metrics for the estimated Hawkes process
            - t_transform (np.ndarray): Transformed event times such that the first observation is at 0 and the last at 1
            - interval_transform (np.ndarray): Transformed inter-event intervals
    """

    # Estimated Hawkes process parameters with the given kernel, baseline and parameters
    hawkes_process = hk.estimator().set_kernel(hwk.KERNEL).set_baseline(hwk.BASELINE)
    hawkes_process.fit(t, [hwk.TIME_ITV_START, hwk.TIME_HORIZON])
    
    # Computed performance metrics for estimated Hawkes process
    metrics = {'Event(s)': np.size(t),
               'Parameters': {k: round(v, 3) for k, v in hawkes_process.para.items()},
               'Branching Ratio': round(hawkes_process.br, 3),
               'Log-Likelihood': round(hawkes_process.L, 3),
               'AIC': round(hawkes_process.AIC, 3)}

    # Written parameters to Parquet file
    write_parquet(metrics, filename=filename)
    # Transformed times so that the first observation is at 0 and the last at 1
    [t_transform, interval_transform] = hawkes_process.t_trans() 
    # Predicted the Hawkes process 
    t_pred = hawkes_process.predict(hwk.END_T, hwk.NUM_SEQ) 

    # Written metrics to a CSV file
    # write_csv(metrics, filename=filename)

    # Plotted the empirical survival function of the estimated Hawkes process (don't work with many iteration)
    # hawkes_process.plot_KS()
    # Plotted the predicted number of events over time (don't work with many iteration)
    # hawkes_process.plot_N_pred()

    return t_pred, metrics, t_transform, interval_transform




