#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Hawkes module

File containing Hawkes process function (simulation/estimation)

"""

from typing import Tuple, TypedDict

import numpy as np
import Hawkes as hk

from UTILS.utils import argparser, write_parquet
import VARIABLES.hawkes_var as hwk

# Simulated Hawkes process 

@argparser(parse_args=False, arg_groups=['hawkes_simulation_params'])
def hawkes_simulation(args_parsed, params: TypedDict = {"mu": 0.1, "alpha": 0.5, "beta": 10.0}) -> Tuple[hk.simulator, np.ndarray]:
    
    """
    Simulated Hawkes process with given parameters

    Args:
        params (TypedDict, optional): Parameters of Hawkes process (default: {"mu": 0.1, "alpha": 0.5, "beta": 10.0})

    Returns:
        Tuple[hk.simulator, np.ndarray]: Hawkes process simulator and the simulated times
    """

    # Parameters initialization
    kernel, baseline, time_itv_start, time_horizon = \
    (args_parsed.kernel, args_parsed.baseline, args_parsed.time_itv_start, args_parsed.time_horizon) \
    if args_parsed else (hwk.KERNEL, hwk.BASELINE, hwk.TIME_ITV_START, hwk.TIME_HORIZON)

    # Created Hawkes process with a given kernel, baseline and parameters
    hawkes_process = hk.simulator().set_kernel(kernel).set_baseline(baseline).set_parameter(params)
    # Simulated Hawkes process in a given time interval
    t = hawkes_process.simulate([time_itv_start, time_horizon])
    
    # Plotted the number of events and intensity over time (don't work with many iteration)
    # hawkes_process.plot_N()
    # hawkes_process.plot_l()

    return hawkes_process, t


# Simulated several Hawkes processes

@argparser(parse_args=False, arg_groups=['hawkes_simulation_params'])
def hawkes_simulations(args_parsed, alpha: np.ndarray, beta: np.ndarray, mu: np.ndarray, filename: str='hawkes_simulations.parquet') -> np.ndarray:
    
    """
    Simulated several Hawkes processes using parameters, and saved results to Parquet file 

    Args:
        alpha (np.ndarray): Excitation matrix of each Hawkes process
        beta (np.ndarray): Decay matrix of each Hawkes process
        mu (np.ndarray): Base intensity of each Hawkes process
        filename (str, optional): Parquet filename to save results (default: "hawkes_simulations.parquet")

    Returns:
        np.ndarray: Simulated event sequences of each Hawkes process
    """

    # Parameters initialization
    process_num, time_horizon = (args_parsed.process_num, args_parsed.time_horizon) if args_parsed else (hwk.PROCESS_NUM, hwk.TIME_HORIZON)

    # Initialized array to store Hawkes processes (Pre-allocate memory)
    simulated_events_seqs = np.zeros((process_num, time_horizon), dtype=np.float32)

    for k in range(process_num):
        # Simulated Hawkes processes with the current simulation parameters
        # The results are stored in the k-th row of the simulated_events_seqs array
        _, t = hawkes_simulation(params={"mu": mu[k], "alpha": alpha[k], "beta": beta[k]})
        
        # Length clipping to not exceed time horizon
        seq_len = np.minimum(len(t), time_horizon)
        simulated_events_seqs[k,:seq_len] = t[:seq_len]
    
    # Written parameters to Parquet file
    write_parquet(simulated_events_seqs, columns=np.arange(time_horizon, dtype=np.int32).astype(str), filename=filename)

    # Created dictionaries list representing simulated event sequences
    # seqs_list = list(map(partial(lambda _, row: {str(idx): x for idx, x in enumerate(row)}, range(time_horizon)), simulated_events_seqs))

    # Written metrics to a CSV file
    # write_csv(seqs_list, filename=filename)

    return simulated_events_seqs


# Estimated Hawkes process

@argparser(parse_args=False, arg_groups=['hawkes_simulation_params'])
def hawkes_estimation(args_parsed, t: np.ndarray, filename: str = "hawkes_estimation.parquet") -> Tuple[np.ndarray, TypedDict, np.ndarray, np.ndarray]:
    
    """
    Estimated Hawkes process from event times, returned predicted process and performance metrics

    Args:
        t (np.ndarray): Event times
        filename (str, optional): Parquet filename for performance metrics (default: "hawkes_estimation.parquet")

    Returns:
        Tuple[np.ndarray, TypedDict, np.ndarray, np.ndarray]: A tuple containing the following items:
            - t_pred (np.ndarray): Predicted event times for estimated Hawkes process
            - metrics (TypedDict): Performance metrics for the estimated Hawkes process
            - t_transform (np.ndarray): Transformed event times such that the first observation is at 0 and the last at 1
            - interval_transform (np.ndarray): Transformed inter-event intervals
    """

    # Parameters initialization
    kernel, baseline, time_itv_start, time_horizon, end_t, num_seq = \
    (args_parsed.kernel, args_parsed.baseline, args_parsed.time_itv_start, args_parsed.time_horizon, args_parsed.end_t, args_parsed.num_seq) \
    if args_parsed else (hwk.KERNEL, hwk.BASELINE, hwk.TIME_ITV_START, hwk.TIME_HORIZON, hwk.END_T, hwk.NUM_SEQ)

    # Estimated Hawkes process parameters with the given kernel, baseline and parameters
    hawkes_process = hk.estimator().set_kernel(kernel).set_baseline(baseline)
    hawkes_process.fit(t, [time_itv_start, time_horizon])
    
    # Computed performance metrics for estimated Hawkes process
    metrics = {'Event(s)': len(t),
               'Parameters': {k: round(v, 3) for k, v in hawkes_process.para.items()},
               'Branching Ratio': round(hawkes_process.br, 3),
               'Log-Likelihood': round(hawkes_process.L, 3),
               'AIC': round(hawkes_process.AIC, 3)}

    # Written parameters to Parquet file
    write_parquet(metrics, filename=filename)
    # Transformed times so that the first observation is at 0 and the last at 1
    [t_transform, interval_transform] = hawkes_process.t_trans() 
    # Predicted the Hawkes process 
    t_pred = hawkes_process.predict(end_t, num_seq) 

    # Written metrics to a CSV file
    # write_csv(metrics, filename=filename)

    # Plotted the empirical survival function of the estimated Hawkes process (don't work with many iteration)
    # hawkes_process.plot_KS()
    # Plotted the predicted number of events over time (don't work with many iteration)
    # hawkes_process.plot_N_pred()

    return t_pred, metrics, t_transform, interval_transform




