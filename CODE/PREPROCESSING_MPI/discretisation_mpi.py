#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Discretisation MPI module

File containing parallelized Aggregated Hawkes Process functions (Hawkes Process discrete conversion).

"""

import numpy as np
from mpi4py import MPI
from functools import partial

import VARIABLES.variables as var
from UTILS.utils import write_csv


# Parallelized jump times histogram for each process (counted number of events which occurred over each interval)

def discretise(jump_times: np.ndarray, filename: str = 'binned_hawkes_simulations.csv') -> np.ndarray:

    # Initialized MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Computed bins number
    num_bins = int(var.TIME_HORIZON // var.DISCRETISE_STEP)

    # Initialized array with dimensions (number of processes, number of jumps per unit of time)
    if rank == 0:
        counts = np.zeros((len(jump_times), num_bins), dtype=np.float64)

    # Pre-allocate memory and scattered data to all processes
    jumps_chunk = np.zeros(len(jump_times) // size, dtype=jump_times.dtype)
    comm.Scatter(jump_times, jumps_chunk, root=0)

    # Computed histogram for each process
    counts_chunk, _ = np.histogram(jumps_chunk, bins=np.linspace(0, var.TIME_HORIZON, num_bins + 1))

    # Gathered results from all processes
    comm.Gather(counts_chunk, counts, root=0)

    if rank == 0:
        # Created dictionaries list representing binned simulated event sequences
        counts_list = list(map(partial(lambda _, row: {str(idx): x for idx, x in enumerate(row)}, range(var.TIME_HORIZON)), counts))

        # Written counts to CSV file
        write_csv(counts_list, filepath=f"{var.FILEPATH}{filename}")

    return counts
