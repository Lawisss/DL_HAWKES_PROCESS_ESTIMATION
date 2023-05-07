#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Hawkes module

File containing Hawkes process function (simulation/estimation)

"""

from typing import Tuple, TypedDict, Optional, Callable

import numpy as np
import polars as pl
import Hawkes as hk
from tqdm import tqdm

import variables.hawkes_var as hwk
from tools.utils import write_parquet


# Simulated Hawkes process 

def hawkes_simulation(params: TypedDict = {"mu": 0.1, "alpha": 0.5, "beta": 10.0}, args: Optional[Callable] = None) -> Tuple[hk.simulator, np.ndarray]:
    
    """
    Simulated Hawkes process with given parameters

    Args:
        params (TypedDict, optional): Parameters of Hawkes process (default: {"mu": 0.1, "alpha": 0.5, "beta": 10.0})
        args (Callable, optional): Arguments if you use run.py instead of tutorial.ipynb (default: None)

    Returns:
        Tuple[hk.simulator, np.ndarray]: Hawkes process simulator and the simulated times
    """

    # Default parameters
    default_params = {"kernel": hwk.KERNEL, 
                      "baseline": hwk.BASELINE, 
                      "time_itv_start": hwk.TIME_ITV_START,
                      "time_horizon": hwk.TIME_HORIZON}

    # Initialized parameters
    dict_args = {k: getattr(args, k, v) for k, v in default_params.items()}

    # Created Hawkes process with a given kernel, baseline and parameters
    hawkes_process = hk.simulator().set_kernel(dict_args['kernel']).set_baseline(dict_args['baseline']).set_parameter(params)
    # Simulated Hawkes process in a given time interval
    t = hawkes_process.simulate([dict_args['time_itv_start'], dict_args['time_horizon']])
    
    # Plotted the number of events and intensity over time (don't work with many iteration)
    # hawkes_process.plot_N()
    # hawkes_process.plot_l()

    return hawkes_process, t


# Simulated several Hawkes processes

def hawkes_simulations(alpha: np.ndarray, beta: np.ndarray, mu: np.ndarray, filename: str='hawkes_simulations.parquet', args: Optional[Callable] = None) -> np.ndarray:
    
    """
    Simulated several Hawkes processes using parameters, and saved results to Parquet file 

    Args:
        alpha (np.ndarray): Excitation matrix of each Hawkes process
        beta (np.ndarray): Decay matrix of each Hawkes process
        mu (np.ndarray): Base intensity of each Hawkes process
        filename (str, optional): Parquet filename to save results (default: "hawkes_simulations.parquet")
        args (Callable, optional): Arguments if you use run.py instead of tutorial.ipynb (default: None)

    Returns:
        np.ndarray: Simulated event sequences of each Hawkes process
    """

    # Default parameters
    default_params = {"process_num": hwk.PROCESS_NUM}

    # Initialized parameters
    dict_args = {k: getattr(args, k, v) for k, v in default_params.items()}

    # Initialized empty dataframe to store simulations
    simulated_events_seqs = pl.DataFrame({"simulations": pl.Series("", dtype=pl.List(pl.Float32))})

    # Started simulations
    with tqdm(total=dict_args['process_num'], desc='Simulation Progress', colour='green') as pbar:

        for i in range(dict_args['process_num']):

            # Simulated Hawkes processes with the current simulation parameters
            _, t = hawkes_simulation(params={"mu": mu[i], "alpha": alpha[i], "beta": beta[i]})
            
            # Listed and concatenated simulations
            series = pl.Series("simulations", [t]).cast(pl.List(pl.Float32))
            simulated_events_seqs = pl.concat([simulated_events_seqs, pl.DataFrame({"simulations": series})], how="vertical")

            # Updated progress bar description
            pbar.set_description(f"Process {i + 1}/{dict_args['process_num']} - Events: {len(t)}")

            # Updated progress bar
            pbar.update(1)
        
    # Written parquet file
    write_parquet(simulated_events_seqs, filename=filename)

    # Created dictionaries list representing simulated event sequences
    # seqs_list = list(map(partial(lambda _, row: {str(idx): x for idx, x in enumerate(row)}, _), simulated_events_seqs))

    # Written metrics to a CSV file
    # write_csv(seqs_list, filename=filename)

    return simulated_events_seqs.to_numpy().tolist()


# Estimated Hawkes process

def hawkes_estimation(t: np.ndarray, filename: str = "hawkes_estimation.parquet", args: Optional[Callable] = None) -> Tuple[np.ndarray, TypedDict, np.ndarray, np.ndarray]:
    
    """
    Estimated Hawkes process from event times, returned predicted process and performance metrics

    Args:
        t (np.ndarray): Event times
        filename (str, optional): Parquet filename for performance metrics (default: "hawkes_estimation.parquet")
        args (Callable, optional): Arguments if you use run.py instead of tutorial.ipynb (default: None)

    Returns:
        Tuple[np.ndarray, TypedDict, np.ndarray, np.ndarray]: A tuple containing the following items:
            - t_pred (np.ndarray): Predicted event times for estimated Hawkes process
            - metrics (TypedDict): Performance metrics for the estimated Hawkes process
            - t_transform (np.ndarray): Transformed event times such that the first observation is at 0 and the last at 1
            - interval_transform (np.ndarray): Transformed inter-event intervals
    """

    # Default parameters
    default_params = {"kernel": hwk.KERNEL, 
                      "baseline": hwk.BASELINE, 
                      "time_itv_start": hwk.TIME_ITV_START,
                      "time_horizon": hwk.TIME_HORIZON,
                      "end_t": hwk.END_T,
                      "num_seq": hwk.NUM_SEQ}

    # Initialized parameters
    dict_args = {k: getattr(args, k, v) for k, v in default_params.items()}

    # Estimated Hawkes process parameters with the given kernel, baseline and parameters
    hawkes_process = hk.estimator().set_kernel(dict_args['kernel']).set_baseline(dict_args['baseline'])
    hawkes_process.fit(t, [dict_args['time_itv_start'], dict_args['time_horizon']])
    
    # Computed performance metrics for estimated Hawkes process
    metrics = {'Event(s)': len(t),
               'Parameters': {k: round(v, 3) for k, v in hawkes_process.para.items()},
               'Branching Ratio': round(hawkes_process.br, 3),
               'Log-Likelihood': round(hawkes_process.L, 3),
               'AIC': round(hawkes_process.AIC, 3)}

    # Written parameters to parquet file
    write_parquet(pl.DataFrame(metrics), filename=filename)
    # Transformed times so that the first observation is at 0 and the last at 1
    [t_transform, interval_transform] = hawkes_process.t_trans() 
    # Predicted the Hawkes process 
    t_pred = hawkes_process.predict(dict_args['end_t'], dict_args['num_seq']) 

    # Written metrics to a CSV file
    # write_csv(metrics, filename=filename)

    # Plotted the empirical survival function of the estimated Hawkes process (don't work with many iteration)
    # hawkes_process.plot_KS()
    # Plotted the predicted number of events over time (don't work with many iteration)
    # hawkes_process.plot_N_pred()

    return t_pred, metrics, t_transform, interval_transform




