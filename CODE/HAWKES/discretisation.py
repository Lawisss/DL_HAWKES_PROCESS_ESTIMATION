#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Discretisation module

File containing aggregated Hawkes process functions (Hawkes process discrete conversion)

"""

import os
from functools import partial
from typing import Optional, Callable

import numpy as np

import VARIABLES.hawkes_var as hwk
from UTILS.utils import write_parquet


# Jump times histogram for each process (counted number of events which occurred over each interval)

def discretise(jump_times: np.ndarray, filename: str = 'binned_hawkes_simulations.parquet', args: Optional[Callable] = None) -> np.ndarray:
    
    """
    Discretized jump times into binned histogram, where bin are time interval of length "hwk.DISCRETISE_STEP"

    Args:
        jump_times (np.ndarray): Jump times for Hawkes process simulation
        filename (str, optional): Filename to write histogram data in Parquet format (default: "binned_hawkes_simulations.parquet")
        args (Callable, optional): Arguments if you use main.py instead of tutorial.ipynb

    Returns:
        np.ndarray: Binned histogram counts for each process, where "num_bins" is number of bins used to discretize jump times
    """  

    # Default parameters
    default_params = {"time_horizon": hwk.TIME_HORIZON, "discretise_step": hwk.DISCRETISE_STEP}

    # Initialized parameters
    dict_args = {k: getattr(args, k, v) for k, v in default_params.items()}

    # Computed bins number
    num_bins = int(dict_args['time_horizon'] // dict_args['discretise_step'])

    # Initialized an array with dimensions (number of processes, number of jumps per unit of time)
    counts = np.zeros((len(jump_times), num_bins), dtype=np.float32)

    # For each process (j), compute jump times histogram (h) using the intervals boundaries specified by the bins
    for j, h in enumerate(jump_times):
        counts[j], _ = np.histogram(h, bins=np.linspace(0, dict_args['time_horizon'], num_bins + 1))

    # Written parameters to Parquet file
    write_parquet(counts, columns=np.arange(dict_args['time_horizon'], dtype=np.int32).astype(str), filename=os.path.join('SIMULATIONS', filename))

    # Created dictionaries list representing binned simulated event sequences
    # counts_list = list(map(partial(lambda _, row: {str(idx): x for idx, x in enumerate(row)}, range(time_horizon)), counts))

    # Written counts to CSV file
    # write_csv(counts_list, filename=filename)
    
    return counts


# Calculated minimum stepsize between events in a given Hawkes process 

def temp_func(jump_times: np.ndarray, args: Optional[Callable] = None) -> float:

    """Calculated minimum step size between events in Hawkes process

    Args:
        jump_times (np.ndarray): Event times in Hawkes process
        args (Callable, optional): Arguments if you use main.py instead of tutorial.ipynb

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


# Calculated temp_func(x, hwk.TIME_HORIZON) minimum for each element x in jump_times

def find_stepsize(jump_times: np.ndarray) -> float:

    """
    Calculated minimum value of "temp_func(x, hwk.TIME_HORIZON)" for each element "x" in "jump_times"

    Args:
        jump_times (np.ndarray): Jump times

    Returns:
        float: Global minimum value of "temp_func(x, hwk.TIME_HORIZON)" for all elements "x" in "jump_times"
    """

    # temp_func computed distance between x and the next value in jump_times
    # Minimum value is the minimum jump time between two successive events 
    return np.min(list(map(temp_func, jump_times)))


# Computed point process jump times from the events history and the time hwk.TIME_HORIZON

def jump_times(h: np.ndarray, args: Optional[Callable] = None) -> np.ndarray:
    
    """
    Computed point process jump times from events history and time horizon

    Args:
        h (np.ndarray): Event history of point process
        args (Callable, optional): Arguments if you use main.py instead of tutorial.ipynb

    Returns:
        np.ndarray: Jump times for point process
    """

    # Default parameters
    default_params = {"time_horizon": hwk.TIME_HORIZON}

    # Initialized parameters
    dict_args = {k: getattr(args, k, v) for k, v in default_params.items()}
    
    # Size of each interval
    stepsize = dict_args["time_horizon"] / len(h)

    # Retrieval of intervals indices with single jump/multiple jumps
    idx_1 = np.nonzero(h == 1)[0]
    idx_2 = np.nonzero(h > 1)[0]

    # Initialized jump times list
    times = np.zeros(len(idx_2) + len(idx_1), dtype=np.float32)

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
