#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Discretisation MPI module

File containing parallelized Aggregated Hawkes process functions (Hawkes process discrete conversion)

"""

from functools import partial
from typing import Optional, Callable, List

import numpy as np
import polars as pl
from mpi4py import MPI

import variables.hawkes_var as hwk
from tools.utils import write_parquet


# Parallelized jump times histogram for each process (counted number of events which occurred over each interval)

def discretise(jump_times: List, root: Optional[int] = 0, filename: Optional[str] = 'binned_hawkes_simulations_mpi.parquet', args: Optional[Callable] = None) -> np.ndarray:

    """
    Discretized parallelized jump times into binned histogram, where bin are time interval of length "hwk.DISCRETISE_STEP"

    Args:
        jump_times (List): Jump times for Hawkes process simulation
        root (int, optional): Rank of process to use as root for MPI communications. (default: 0)
        filename (str, optional): Filename to write histogram data in parquet format (default: "binned_hawkes_simulations_mpi.parquet")
        args (Callable, optional): Arguments if you use run.py instead of tutorial.ipynb (default: None)

    Returns:
        np.ndarray: Binned histogram counts for each process, where "num_bins" is number of bins used to discretize jump times
    """  

    # Initialized MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Default parameters
    default_params = {"time_horizon": hwk.TIME_HORIZON, "discretise_step": hwk.DISCRETISE_STEP}

    # Initialized parameters
    dict_args = {k: getattr(args, k, v) for k, v in default_params.items()}

    # Computed bins number
    num_bins = int(dict_args['time_horizon'] // dict_args['discretise_step'])

    # Initialized array with dimensions (number of processes, number of jumps per unit of time)
    if rank == 0:
        counts = np.zeros((len(jump_times), num_bins), dtype=np.float32)

    # Pre-allocated memory and scattered data to all processes
    jumps_chunk = np.zeros(len(jump_times) // size, dtype=jump_times.dtype)
    comm.Scatter(jump_times, jumps_chunk, root=root)

    # Computed histogram for each process
    counts_chunk, _ = np.histogram(jumps_chunk, bins=np.linspace(0, dict_args['time_horizon'], num_bins + 1))

    # Gathered results from all processes
    comm.Gather(counts_chunk, counts, root=root)

    # Written parameters to parquet file
    if rank == 0:
        write_parquet(pl.DataFrame(counts, schema=np.arange(dict_args['time_horizon'], dtype=np.int32).astype(str).tolist()), filename=filename)

        # Created dictionaries list representing binned simulated event sequences
        # counts_list = list(map(partial(lambda _, row: {str(idx): x for idx, x in enumerate(row)}, range(hwk.TIME_HORIZON)), counts))
        # Written counts to CSV file
        # write_csv(counts_list, filename=filename)

    return counts


# Calculated minimum stepsize between events in a given Hawkes process 

def temp_func(jump_times: np.ndarray, args: Optional[Callable] = None) -> float:

    """Calculated minimum step size between events in Hawkes process

    Args:
        jump_times (np.ndarray): Event times in Hawkes process
        args (Callable, optional): Arguments if you use run.py instead of tutorial.ipynb (default: None)

    Returns:
        float: Minimum step size between events in Hawkes process. If no events, step size is set to maximum time horizon
    """    

    # Default parameters
    default_params = {"time_horizon": hwk.TIME_HORIZON}

    # Initialized parameters
    dict_args = {k: getattr(args, k, v) for k, v in default_params.items()}

    # If no event has been recorded, step size = hwk.TIME_HORIZON
    if len(jump_times) == 0:
        stepsize = dict_args["time_horizon"] 

    else:
        # Added event times and boundaries
        times = np.concatenate(([0], jump_times, [dict_args["time_horizon"]]))  
        # Calculated the differences between the times
        diff = np.diff(times)  
        # Removed negative differences
        diff = diff[diff > 0]  
        # Taken the smallest positive difference
        stepsize = partial(np.around, decimals=1)(np.min(diff))    

    return stepsize



# Calculated parallelized temp_func(x, hwk.TIME_HORIZON) minimum for each element x in jump_times

def find_stepsize(jump_times: np.ndarray, root: Optional[int] = 0) -> float:

    """
    Calculated parallelized minimum value of "temp_func(x, hwk.TIME_HORIZON)" for each element "x" in "jump_times"

    Args:
        jump_times (np.ndarray): Jump times
        root (int, optional): Rank of root process (default: 0)

    Returns:
        float: Global minimum value of "temp_func(x, hwk.TIME_HORIZON)" for all elements "x" in "jump_times"
    """

    # Initialized MPI
    comm = MPI.COMM_WORLD
    _ = comm.Get_rank()
    size = comm.Get_size()

    # Scattered data across processes
    chunk = np.zeros(len(jump_times) // size, dtype=np.float32)
    comm.Scatter(jump_times, chunk, root=root)

    # Computed chunk minimum value
    chunk_min = np.min(list(map(temp_func, chunk)))

    # Reduced chunk minimum values to global minimum value
    global_min = np.zeros(1, dtype=np.float32)
    comm.Reduce(chunk_min, global_min, op=MPI.MIN, root=root)

    # Broadcast global minimum value to all processes
    comm.Bcast(global_min, root=root)

    return global_min[0]


# Computed parallelized point process jump times from the events history and the time hwk.TIME_HORIZON

def jump_times(h: np.ndarray, root: Optional[int] = 0, args: Optional[Callable] = None) -> np.ndarray:

    """
    Computed parallelized point process jump times from events history and time horizon

    Args:
        h (np.ndarray): Event history of point process
        root (int, optional): Rank of root process for gathering results (default: 0)
        args (Callable, optional): Arguments if you use run.py instead of tutorial.ipynb (default: None)

    Returns:
        np.ndarray: Jump times for point process
    """
    
    # Initialized MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Default parameters
    default_params = {"discretise_step": hwk.DISCRETISE_STEP}

    # Initialized parameters
    dict_args = {k: getattr(args, k, v) for k, v in default_params.items()}

    # Indices with single jump/multiple jumps
    times = []
    idx_1 = np.where(h == 1)[0]
    idx_2 = np.where(h > 1)[0]
    stepsize = dict_args["discretise_step"]

    # Generation of jump times
    if len(idx_2) > 0:
        repeat_idx = np.repeat(idx_2, h[idx_2].astype(int))
        times.extend(np.random.uniform(repeat_idx * stepsize, (repeat_idx + 1) * stepsize))

    # Added and sorted jump times (intervals with single jump)
    times.extend((idx_1 * stepsize) - (0.5 * stepsize))
    jump_times = np.sort(times)

    # Gathered results from all processors
    jump_times = comm.gather(jump_times, root=root)

    # Concatenate/sort the jump times
    if rank == 0:
        jump_times = np.concatenate(jump_times)
        jump_times.sort()

    return jump_times


