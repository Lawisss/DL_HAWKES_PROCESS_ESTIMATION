#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Discretisation MPI module

File containing parallelized Aggregated Hawkes process functions (Hawkes process discrete conversion).

"""

import numpy as np
from mpi4py import MPI
from functools import partial

import VARIABLES.hawkes_var as hwk
from UTILS.utils import write_csv


# Parallelized jump times histogram for each process (counted number of events which occurred over each interval)

def discretise(jump_times: np.ndarray, root: int = 0, filename: str = 'binned_hawkes_simulations.csv') -> np.ndarray:

    # Initialized MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Computed bins number
    num_bins = int(hwk.TIME_HORIZON // hwk.DISCRETISE_STEP)

    # Initialized array with dimensions (number of processes, number of jumps per unit of time)
    if rank == 0:
        counts = np.zeros((len(jump_times), num_bins), dtype=np.float32)

    # Pre-allocated memory and scattered data to all processes
    jumps_chunk = np.zeros(len(jump_times) // size, dtype=jump_times.dtype)
    comm.Scatter(jump_times, jumps_chunk, root=root)

    # Computed histogram for each process
    counts_chunk, _ = np.histogram(jumps_chunk, bins=np.linspace(0, hwk.TIME_HORIZON, num_bins + 1))

    # Gathered results from all processes
    comm.Gather(counts_chunk, counts, root=root)

    if rank == 0:
        # Created dictionaries list representing binned simulated event sequences
        counts_list = list(map(partial(lambda _, row: {str(idx): x for idx, x in enumerate(row)}, range(hwk.TIME_HORIZON)), counts))

        # Written counts to CSV file
        write_csv(counts_list, filename=filename)

    return counts


# Calculated minimum stepsize between events in a given Hawkes process 

def temp_func(jump_times: np.ndarray) -> float:

    # If no event has been recorded, step size = hwk.TIME_HORIZON
    if len(jump_times) == 0:
        stepsize = hwk.TIME_HORIZON 

    else:
        # Added event times and boundaries
        times = np.concatenate(([0], jump_times, [hwk.TIME_HORIZON]))  
        # Calculated the differences between the times
        diff = np.diff(times)  
        # Removed negative differences
        diff = diff[diff > 0]  
        # Taken the smallest positive difference
        stepsize = partial(np.around, decimals=1)(np.min(diff))    

    return stepsize



# Calculated parallelized temp_func(x, hwk.TIME_HORIZON) minimum for each element x in jump_times

def find_stepsize(jump_times: np.ndarray, root: int = 0) -> float:

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

def jump_times(h: np.ndarray, root: int = 0) -> np.ndarray:

    # Initialized MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Size of each interval
    stepsize = hwk.TIME_HORIZON / len(h)

    # Retrieval of intervals indices with single jump/multiple jumps
    idx_1 = np.nonzero(h == 1)[0]
    idx_2 = np.nonzero(h > 1)[0]

    # Divided indices into chunks
    chunks = np.array_split(idx_2, size)

    # Initialized jump times list
    times_chunk = np.zeros(len(idx_2) // size + len(idx_1), dtype=np.float32)

    # Intervals with multiple jumps
    if len(idx_2) > 0:
        # Process chunks in parallel
        for i in chunks[rank]:
            # Jumps number in i
            n_jumps = h[i]

            # Bounds of i
            t_start = (i - 1) * stepsize
            t_end = i * stepsize

            # Generation of jump times for i
            times_chunk[i:i+n_jumps] = np.random.uniform(t_start, t_end, size=(n_jumps,))

    # Intervals with a single jump
    if len(idx_1) > 0:
        times_chunk[len(idx_2) // size:] = idx_1 * stepsize - 0.5 * stepsize

    # Gathered results from all processors
    jump_times = comm.gather(times_chunk, root=root)

    # Concatenate/sort the jump times
    if rank == 0:
        jump_times = np.concatenate(jump_times)
        jump_times.sort()

    return jump_times


