#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Hawkes MPI module

File containing parallelized Hawkes Process function (simulation/estimation).

"""

import numpy as np
import Hawkes as hk
from mpi4py import MPI
from functools import partial
from typing import Tuple, TypedDict

from UTILS.utils import write_csv
import VARIABLES.variables as var

# Parallelized simulated Hawkes process 

def hawkes_simulation(params: TypedDict = {"mu": 0.1, "alpha": 0.5, "beta": 10.0}) -> Tuple[hk.simulator, np.ndarray]:

    # Initialized MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Broadcast the parameters to all processes
    params = comm.bcast(params, root=0)

    # Divide the time horizon into equal intervals for each process
    time_intervals = np.linspace(var.TIME_ITV_START, var.TIME_HORIZON, size + 1)

    # Determine the start and end time interval for the current process
    start_time = time_intervals[rank]
    end_time = time_intervals[rank + 1]

    # Create a Hawkes process with the given kernel, baseline and parameters
    hawkes_process = hk.simulator().set_kernel(var.KERNEL).set_baseline(var.BASELINE).set_parameter(params)

    # Simulate a Hawkes process in the given time interval
    T = hawkes_process.simulate([start_time, end_time])

    # Gather the results from all processes
    T_all = comm.gather(T, root=0)

    # Concatenate the results into a single array
    if rank == 0:
        T = np.concatenate(T_all)

    # Return the Hawkes process and the simulated times
    return hawkes_process, T


# Parallelized simulated Hawkes processes

def hawkes_simulations(mu: np.ndarray, alpha: np.ndarray, beta: np.ndarray, filename: str='hawkes_simulations.csv') -> np.ndarray:
    
    # Initialized MPI environment
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # Splitted process number evenly among MPI ranks
    local_process_num = var.PROCESS_NUM // size

    # Generated simulation parameters for each local process
    local_mu = mu[rank * local_process_num:(rank + 1) * local_process_num]
    local_alpha = alpha[rank * local_process_num:(rank + 1) * local_process_num]
    local_beta = beta[rank * local_process_num:(rank + 1) * local_process_num]

    # Initialize a filled with zeros array to store Hawkes processes (Pre-allocate memory)
    simulated_events_seqs = np.zeros((local_process_num, var.TIME_HORIZON), dtype=np.float64)

    for k in range(local_process_num):
        # Simulate a Hawkes processes with the current simulation parameters
        # The results are stored in the k-th row of the simulated_events_seqs array
        _, T = hawkes_simulation(params={"mu": local_mu[k], "alpha": local_alpha[k], "beta": local_beta[k]})
        
        # Convert temporary list T to an array and store the results in simulated_events_seqs
        simulated_events_seqs[k,:] = np.asarray(T)[:var.TIME_HORIZON]

    # Gather simulated_events_seqs arrays from all ranks to root
    simulated_events_seqs = comm.gather(simulated_events_seqs, root=0)

    if rank == 0:
        # Concatenated simulated_events_seqs arrays into single array
        simulated_events_seqs = np.concatenate(simulated_events_seqs)

        # Created list of dictionaries representing the simulated event sequences
        seqs_list = list(map(partial(lambda _, row: {str(idx): x for idx, x in enumerate(row)}, range(var.TIME_HORIZON)), simulated_events_seqs))

        # Written metrics to a CSV file
        write_csv(seqs_list, filepath=f"{var.FILEPATH}{filename}")

    return simulated_events_seqs