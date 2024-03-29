#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Hawkes MPI module

File containing parallelized Hawkes process function (simulation/estimation)

"""

from functools import partial
from typing import Tuple, TypedDict, Optional, Callable

import numpy as np
import polars as pl
import Hawkes as hk
from mpi4py import MPI

import variables.hawkes_var as hwk
from tools.utils import write_parquet



# Parallelized simulated Hawkes process 

def hawkes_simulation(params: Optional[TypedDict] = {"mu": 0.1, "alpha": 0.5, "beta": 10.0}, root: Optional[int] = 0, args: Optional[Callable] = None) -> Tuple[hk.simulator, np.ndarray]:

    """
    Simulated parallelized Hawkes process with given parameters

    Args:
        params (TypedDict, optional): Parameters of Hawkes process (default: {"mu": 0.1, "alpha": 0.5, "beta": 10.0})
        root (int, optional): Rank of process to use as root for MPI communications (default: 0)
        args (Callable, optional): Arguments if you use run.py instead of tutorial.ipynb (default: None)

    Returns:
        Tuple[hk.simulator, np.ndarray]: Hawkes process simulator and the simulated times
    """
    
    # Initialized MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Default parameters
    default_params = {"kernel": hwk.KERNEL, 
                      "baseline": hwk.BASELINE, 
                      "time_itv_start": hwk.TIME_ITV_START,
                      "time_horizon": hwk.TIME_HORIZON}

    # Initialized parameters
    dict_args = {k: getattr(args, k, v) for k, v in default_params.items()}

    # Broadcast parameters to all processes
    params = comm.bcast(params, root=root)

    # Divided time horizon into equal intervals for each process
    time_intervals = np.linspace(dict_args['time_itv_start'], dict_args['time_horizon'], size + 1)

    # Determined the start / end time interval for current process
    start_time = time_intervals[rank]
    end_time = time_intervals[rank + 1]

    # Created Hawkes process with kernel, baseline and parameters
    hawkes_process = hk.simulator().set_kernel(dict_args['kernel']).set_baseline(dict_args['baseline']).set_parameter(params)

    # Simulated a Hawkes process in time interval
    t = hawkes_process.simulate([start_time, end_time])

    # Gathered results from all processes
    t_processes = comm.gather(t, root=root)

    # Concatenated results
    if rank == 0:
        t = np.concatenate(t_processes)

    return hawkes_process, t


# Parallelized simulated Hawkes processes

def hawkes_simulations(alpha: np.ndarray, beta: np.ndarray, mu: np.ndarray, root: Optional[int] = 0, filename: Optional[str] = 'hawkes_simulations_mpi.parquet', args: Optional[Callable] = None) -> np.ndarray:
    
    """
    Simulated several parallelized Hawkes processes using parameters, and saved results to parquet file 

    Args:
        alpha (np.ndarray): Excitation matrix of each Hawkes process
        beta (np.ndarray): Decay matrix of each Hawkes process
        mu (np.ndarray): Base intensity of each Hawkes process
        root (int, optional): Rank of process to use as root for MPI communications (default: 0)
        filename (str, optional): Parquet filename to save results (default: "hawkes_simulations_mpi.parquet")
        args (Callable, optional): Arguments if you use run.py instead of tutorial.ipynb (default: None)

    Returns:
        np.ndarray: Simulated event sequences of each Hawkes process
    """
    
    # Initialized MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # Default parameters
    default_params = {"process_num": hwk.PROCESS_NUM}

    # Initialized parameters
    dict_args = {k: getattr(args, k, v) for k, v in default_params.items()}

    # Splitted process number evenly among MPI ranks
    chunk_num = dict_args['process_num'] // size

    # Generated simulation parameters for each local process
    chunk_mu = mu[rank * chunk_num:(rank + 1) * chunk_num]
    chunk_alpha = alpha[rank * chunk_num:(rank + 1) * chunk_num]
    chunk_beta = beta[rank * chunk_num:(rank + 1) * chunk_num]

    # Initialized array to store Hawkes processes (Pre-allocate memory)
    simulated_events_seqs = np.zeros((dict_args['process_num'], ))

    for k in range(chunk_num):
        # Simulated Hawkes processes with current simulation parameters
        # The results are stored in the k-th row of the simulated_events_seqs
        _, t = hawkes_simulation(params={"mu": chunk_mu[k], "alpha": chunk_alpha[k], "beta": chunk_beta[k]})

        simulated_events_seqs[k] = t

    # Gather simulated_events_seqs from all ranks to root
    simulated_events_seqs = comm.gather(simulated_events_seqs, root=root)

    if rank == 0:
        # Concatenated simulated_events_seqs
        simulated_events_seqs = np.concatenate(simulated_events_seqs)

        # Written parameters to parquet file
        write_parquet(pl.DataFrame(simulated_events_seqs), filename=filename)

        # Created dictionaries list representing simulated event sequences
        # seqs_list = list(map(partial(lambda _, row: {str(idx): x for idx, x in enumerate(row)}, range(hwk.TIME_HORIZON)), simulated_events_seqs))
        # Written metrics to a CSV file
        # write_csv(seqs_list, filename=filename)

    return simulated_events_seqs


# Parallelized estimated Hawkes process

def hawkes_estimation(t: np.ndarray, root: Optional[int] = 0, filename: Optional[str] = "hawkes_estimation_mpi.parquet", args: Optional[Callable] = None) -> Tuple[np.ndarray, TypedDict, np.ndarray, np.ndarray]:
    
    """
    Estimated parallelized Hawkes process from event times and returns predicted process and performance metrics

    Args:
        t (np.ndarray): Event times
        root (int, optional): Rank of process to use as root for MPI communications. (default: 0)
        filename (str, optional): Parquet filename for performance metrics. (default: "hawkes_estimation_mpi.parquet")
        args (Callable, optional): Arguments if you use run.py instead of tutorial.ipynb (default: None)

    Returns:
        Tuple[np.ndarray, TypedDict, np.ndarray, np.ndarray]: A tuple containing the following items:
            - t_pred (np.ndarray): Predicted event times for estimated Hawkes process
            - metrics (TypedDict): Performance metrics for the estimated Hawkes process
            - t_transform (np.ndarray): Transformed event times such that the first observation is at 0 and the last at 1
            - interval_transform (np.ndarray): Transformed inter-event intervals
    """

    # Initialized MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Default parameters
    default_params = {"kernel": hwk.KERNEL, 
                      "baseline": hwk.BASELINE, 
                      "time_itv_start": hwk.TIME_ITV_START,
                      "time_horizon": hwk.TIME_HORIZON,
                      "end_t": hwk.END_T,
                      "num_seq": hwk.NUM_SEQ}

    # Initialized parameters
    dict_args = {k: getattr(args, k, v) for k, v in default_params.items()}

    # Estimated Hawkes process parameters with kernel, baseline and parameters
    hawkes_process = hk.estimator().set_kernel(dict_args['kernel']).set_baseline(dict_args['baseline'])

    # Broadcast parameters
    params = comm.bcast([dict_args['time_itv_start'], dict_args['time_horizon'], dict_args['end_t'], dict_args['num_seq']], root=root)

    # Distributed among processes
    t_chunks = np.array_split(t, size)
    t_chunk = comm.scatter(t_chunks, root=root)

    # Fitted process on each chunk
    hawkes_process.fit(t_chunk, params)
    t_pred_chunk = hawkes_process.predict(params[2], params[3])

    # Gathered all predicted chunks
    t_pred_chunks = comm.gather(t_pred_chunk, root=root)

    if rank == 0:
        # Concatenated predictions
        t_pred = np.concatenate(t_pred_chunks)

        # Computed performance metrics for the estimated Hawkes process
        metrics = {'Event(s)': len(t),
                   'Parameters': {k: round(v, 3) for k, v in hawkes_process.para.items()},
                   'Branching Ratio': round(hawkes_process.br, 3),
                   'Log-Likelihood': round(hawkes_process.L, 3),
                   'AIC': round(hawkes_process.AIC, 3)}
        
        # Written parameters to parquet file
        write_parquet(pl.DataFrame(metrics), filename=filename)
        # Transformed times so that the first observation is at 0 and the last at 1
        [t_transform, interval_transform] = hawkes_process.t_trans() 

        # Written metrics to a CSV file
        # write_csv(metrics, filename=filename)

        return t_pred, metrics, t_transform, interval_transform
