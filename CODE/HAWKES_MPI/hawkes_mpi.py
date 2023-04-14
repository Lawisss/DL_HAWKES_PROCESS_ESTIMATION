#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Hawkes MPI module

File containing parallelized Hawkes process function (simulation/estimation).

"""

from functools import partial
from typing import Tuple, TypedDict

import numpy as np
import Hawkes as hk
from mpi4py import MPI

from UTILS.utils import write_csv
import VARIABLES.hawkes_var as hwk


# Parallelized simulated Hawkes process 

def hawkes_simulation(params: TypedDict = {"mu": 0.1, "alpha": 0.5, "beta": 10.0}, root: int = 0) -> Tuple[hk.simulator, np.ndarray]:

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
    T = hawkes_process.simulate([start_time, end_time])

    # Gathered results from all processes
    T_processes = comm.gather(T, root=root)

    # Concatenated results
    if rank == 0:
        T = np.concatenate(T_processes)

    return hawkes_process, T


# Parallelized simulated Hawkes processes

def hawkes_simulations(mu: np.ndarray, alpha: np.ndarray, beta: np.ndarray, root: int = 0, filename: str='hawkes_simulations.csv') -> np.ndarray:
    
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
        _, T = hawkes_simulation(params={"mu": chunk_mu[k], "alpha": chunk_alpha[k], "beta": chunk_beta[k]})
        
        # Converted temporary list T to array and stored results in simulated_events_seqs
        simulated_events_seqs[k,:] = np.asarray(T)[:hwk.TIME_HORIZON]

    # Gather simulated_events_seqs from all ranks to root
    simulated_events_seqs = comm.gather(simulated_events_seqs, root=root)

    if rank == 0:
        # Concatenated simulated_events_seqs
        simulated_events_seqs = np.concatenate(simulated_events_seqs)

        # Created dictionaries list representing simulated event sequences
        seqs_list = list(map(partial(lambda _, row: {str(idx): x for idx, x in enumerate(row)}, range(hwk.TIME_HORIZON)), simulated_events_seqs))

        # Written metrics to a CSV file
        write_csv(seqs_list, filename=filename)

    return simulated_events_seqs


# Estimated Hawkes process

def hawkes_estimation(T: np.ndarray, root: int = 0, filename: str = "hawkes_estimation.csv") -> Tuple[np.ndarray, TypedDict, np.ndarray, np.ndarray]:
    
    # Initialized MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Estimated Hawkes process parameters with kernel, baseline and parameters
    hawkes_process = hk.estimator().set_kernel(hwk.KERNEL).set_baseline(hwk.BASELINE)

    # Broadcast parameters
    params = comm.bcast([hwk.TIME_ITV_START, hwk.TIME_HORIZON, hwk.END_T, hwk.NUM_SEQ], root=root)

    # Distributed among processes
    T_chunks = np.array_split(T, size)
    T_chunk = comm.scatter(T_chunks, root=root)

    # Fitted process on each chunk
    hawkes_process.fit(T_chunk, params)
    T_pred_chunk = hawkes_process.predict(params[2], params[3])

    # Gathered all predicted chunks
    T_pred_chunks = comm.gather(T_pred_chunk, root=root)

    if rank == 0:
        # Concatenated predictions
        T_pred = np.concatenate(T_pred_chunks)

        # Computed performance metrics for the estimated Hawkes process
        metrics = {'Event(s)': len(T),
                   'Parameters': {k: round(v, 3) for k, v in hawkes_process.para.items()},
                   'Branching Ratio': round(hawkes_process.br, 3),
                   'Log-Likelihood': round(hawkes_process.L, 3),
                   'AIC': round(hawkes_process.AIC, 3)}
        
        # Written metrics to a CSV file
        write_csv(metrics, filename=filename)

        # Transformed times so that the first observation is at 0 and the last at 1
        [T_transform, interval_transform] = hawkes_process.t_trans() 
        
        return T_pred, metrics, T_transform, interval_transform
