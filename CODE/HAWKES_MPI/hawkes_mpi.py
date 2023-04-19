#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Hawkes MPI module

File containing parallelized Hawkes process function (simulation/estimation)

"""

from functools import partial
from typing import Tuple, TypedDict

import numpy as np
import Hawkes as hk
from mpi4py import MPI

from UTILS.utils import write_parquet
import VARIABLES.hawkes_var as hwk


# Parallelized simulated Hawkes process 

def hawkes_simulation(params: TypedDict = {"mu": 0.1, "alpha": 0.5, "beta": 10.0}, root: int = 0) -> Tuple[hk.simulator, np.ndarray]:

    """
    Simulated parallelized Hawkes process with given parameters

    Args:
        params (TypedDict, optional): Parameters of Hawkes process. Default is {"mu": 0.1, "alpha": 0.5, "beta": 10.0}
        root (int): Rank of process to use as root for MPI communications. Default is 0

    Returns:
        Tuple[hk.simulator, np.ndarray]: Hawkes process simulator and the simulated times
    """
    
    # Initialized MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Broadcast parameters to all processes
    params = comm.bcast(params, root=root)

    # Divided time horizon into equal intervals for each process
    time_intervals = np.linspace(hwk.TIME_ITV_START, hwk.TIME_HORIZON, size + 1)

    # Determined the start / end time interval for current process
    start_time = time_intervals[rank]
    end_time = time_intervals[rank + 1]

    # Created Hawkes process with kernel, baseline and parameters
    hawkes_process = hk.simulator().set_kernel(hwk.KERNEL).set_baseline(hwk.BASELINE).set_parameter(params)

    # Simulated a Hawkes process in time interval
    t = hawkes_process.simulate([start_time, end_time])

    # Gathered results from all processes
    t_processes = comm.gather(t, root=root)

    # Concatenated results
    if rank == 0:
        t = np.concatenate(t_processes)

    return hawkes_process, t


# Parallelized simulated Hawkes processes

def hawkes_simulations(alpha: np.ndarray, beta: np.ndarray, mu: np.ndarray, root: int = 0, filename: str='hawkes_simulations_mpi.parquet') -> np.ndarray:
    
    """
    Simulated several parallelized Hawkes processes using parameters, and saved results to Parquet file 

    Args:
        alpha (np.ndarray): Excitation matrix of each Hawkes process
        beta (np.ndarray): Decay matrix of each Hawkes process
        mu (np.ndarray): Base intensity of each Hawkes process
        root (int): Rank of process to use as root for MPI communications. Default is 0
        filename (str, optional): Parquet filename to save results. Defaults is 'hawkes_simulations_mpi.parquet'

    Returns:
        np.ndarray: Simulated event sequences of each Hawkes process
    """
    
    # Initialized MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # Splitted process number evenly among MPI ranks
    chunk_num = hwk.PROCESS_NUM // size

    # Generated simulation parameters for each local process
    chunk_mu = mu[rank * chunk_num:(rank + 1) * chunk_num]
    chunk_alpha = alpha[rank * chunk_num:(rank + 1) * chunk_num]
    chunk_beta = beta[rank * chunk_num:(rank + 1) * chunk_num]

    # Initialized array to store Hawkes processes (Pre-allocate memory)
    simulated_events_seqs = np.zeros((chunk_num, hwk.TIME_HORIZON), dtype=np.float32)

    for k in range(chunk_num):
        # Simulated Hawkes processes with current simulation parameters
        # The results are stored in the k-th row of the simulated_events_seqs
        _, t = hawkes_simulation(params={"mu": chunk_mu[k], "alpha": chunk_alpha[k], "beta": chunk_beta[k]})
        
        # Length clipping to not exceed time horizon
        seq_len = np.minimum(len(t), hwk.TIME_HORIZON)
        simulated_events_seqs[k,:seq_len] = t[:seq_len]

    # Gather simulated_events_seqs from all ranks to root
    simulated_events_seqs = comm.gather(simulated_events_seqs, root=root)

    if rank == 0:
        # Concatenated simulated_events_seqs
        simulated_events_seqs = np.concatenate(simulated_events_seqs)

        # Written parameters to Parquet file
        write_parquet(simulated_events_seqs, columns=np.arange(hwk.TIME_HORIZON, dtype=np.int32).astype(str), filename=filename)

        # Created dictionaries list representing simulated event sequences
        # seqs_list = list(map(partial(lambda _, row: {str(idx): x for idx, x in enumerate(row)}, range(hwk.TIME_HORIZON)), simulated_events_seqs))
        # Written metrics to a CSV file
        # write_csv(seqs_list, filename=filename)

    return simulated_events_seqs


# Parallelized estimated Hawkes process

def hawkes_estimation(t: np.ndarray, root: int = 0, filename: str = "hawkes_estimation_mpi.parquet") -> Tuple[np.ndarray, TypedDict, np.ndarray, np.ndarray]:
    
    """
    Estimated parallelized Hawkes process from event times and returns predicted process and performance metrics

    Args:
        t (np.ndarray): Event times
        root (int): Rank of process to use as root for MPI communications. Default is 0
        filename (str, optional): Parquet filename for performance metrics. Defaults is "hawkes_estimation_mpi.parquet"

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

    # Estimated Hawkes process parameters with kernel, baseline and parameters
    hawkes_process = hk.estimator().set_kernel(hwk.KERNEL).set_baseline(hwk.BASELINE)

    # Broadcast parameters
    params = comm.bcast([hwk.TIME_ITV_START, hwk.TIME_HORIZON, hwk.END_T, hwk.NUM_SEQ], root=root)

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
        
        # Written parameters to Parquet file
        write_parquet(metrics, filename=filename)
        # Transformed times so that the first observation is at 0 and the last at 1
        [t_transform, interval_transform] = hawkes_process.t_trans() 

        # Written metrics to a CSV file
        # write_csv(metrics, filename=filename)

        return t_pred, metrics, t_transform, interval_transform
