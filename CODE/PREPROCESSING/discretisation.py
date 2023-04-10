#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Discretisation module

File containing Aggregated Hawkes Process functions (Hawkes Process discrete conversion).

"""

import numpy as np
from functools import partial

import VARIABLES.variables as var
from UTILS.utils import write_csv


# Jump times histogram for each process (counted number of events which occurred over each interval)

def discretise(jump_times: np.ndarray, filename: str = 'binned_hawkes_simulations.csv') -> np.ndarray:

    # Computed bins number
    num_bins = int(var.TIME_HORIZON // var.DISCRETISE_STEP)

    # Initialized an array with dimensions (number of processes, number of jumps per unit of time)
    counts = np.zeros((len(jump_times), num_bins), dtype=np.float64)

    # For each process (j), compute jump times histogram (h) using the intervals boundaries specified by the bins
    for j, h in enumerate(jump_times):
        counts[j], _ = np.histogram(h, bins=np.linspace(0, var.TIME_HORIZON, num_bins + 1))

    # Created dictionaries list representing binned simulated event sequences
    counts_list = list(map(partial(lambda _, row: {str(idx): x for idx, x in enumerate(row)}, range(var.TIME_HORIZON)), counts))

    # Written counts to CSV file
    write_csv(counts_list, filename=filename)

    return counts


# Calculated minimum stepsize between events in a given Hawkes process 

def temp_func(jump_times: np.ndarray) -> float:

    # If no event has been recorded, step size = svar.TIME_HORIZON
    if len(jump_times) == 0:
        stepsize = var.TIME_HORIZON 

    else:
        # Added event times and boundaries
        times = np.concatenate(([0], jump_times, [var.TIME_HORIZON]))  
        # Calculated the differences between the times
        diff = np.diff(times)  
        # Removed negative differences
        diff = diff[diff > 0]  
        # Taken the smallest positive difference
        stepsize = partial(np.around, decimals=1)(np.min(diff))    

    return stepsize


# Calculated temp_func(x, var.TIME_HORIZON) minimum for each element x in jump_times

def find_stepsize(jump_times: np.ndarray) -> float:
    # temp_func computed distance between x and the next value in jump_times
    # Minimum value is the minimum jump time between two successive events 
    return np.min(list(map(temp_func, jump_times)))


# Computed jump times of point process from the events history and the time var.TIME_HORIZON

def jump_times(h: np.ndarray) -> np.ndarray:
    # Size of each interval
    stepsize = var.TIME_HORIZON / len(h)

    # Retrieval of intervals indices with single jump/multiple jumps
    idx_1 = np.nonzero(h == 1)[0]
    idx_2 = np.nonzero(h > 1)[0]

    # Initialized jump times list
    times = np.zeros(len(idx_2) + len(idx_1), dtype=np.float64)

    # Variable to track the index of the times list
    k = 0

    # Intervals with multiple jumps
    if len(idx_2) > 0:
        for i in idx_2:
            # Jumps number in i
            n_jumps = h[i]

            # Bounds of i
            t_start = (i - 1) * stepsize
            t_end = i * stepsize

            # Generation of jump times for i
            times[k:k+n_jumps] = np.random.uniform(t_start, t_end, size=(n_jumps,))

            # Update times list index
            k += n_jumps

    # Intervals with a single jump
    if len(idx_1) > 0:
        times[k:] = idx_1 * stepsize - 0.5 * stepsize

    # Lists concatenation and jump times sorted
    jump_times = np.concatenate([times])
    jump_times.sort()

    return jump_times
