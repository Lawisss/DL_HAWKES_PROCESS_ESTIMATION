#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Discretisation module

File containing Aggregated Hawkes Process functions (Hawkes Process conversion).

"""

import numpy as np
import pandas as pd

import VARIABLES.variables as var

# Computed the histogram of jump times for each process (counted number of events which occurred over each interval)

def discretise(jump_times, filename='binned_hawkes_simulations.csv'):

    # Computed bins number
    num_bins = int(var.TIME_HORIZON // var.DISCRETISE_STEP)

    # Initialized an array of zeros with dimensions (number of processes, number of jumps per unit of time)
    counts = np.zeros((len(jump_times), num_bins))

    # For each process (j), compute jump times histogram (h) using the intervals boundaries specified by the bins
    for j, h in enumerate(jump_times):
        counts[j], _ = np.histogram(h, bins=np.linspace(0, var.TIME_HORIZON, num_bins + 1))
                                    
    # Created a DataFrame, name the columns, and generate csv file
    df = pd.DataFrame(np.row_stack(counts))
    df.to_csv(f"{var.FILEPATH}{filename}", index=False)

    return counts


# Calculated minimum stepsize between events in a given Hawkes process 

def temp_func(jump_times):

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
        # Taken the smallest positive difference and round to 1 decimal
        stepsize = np.around(np.min(diff), 1)  

    return stepsize


# Calculated temp_func(x, var.TIME_HORIZON) minimum for each element x in jump_times

def find_stepsize(jump_times):
    # temp_func computed distance between x and the next value in jump_times
    # Minimum value is the minimum jump time between two successive events 
    return np.min([temp_func(x, var.TIME_HORIZON) for x in jump_times])


# Computed jump times of point process from the events history and the time var.TIME_HORIZON

def jump_times(h):
    # Size of each interval
    stepsize = var.TIME_HORIZON / len(h)

    # Retrieval of intervals indices with single jump/multiple jumps
    idx_1 = np.nonzero(h == 1)[0]
    idx_2 = np.nonzero(h > 1)[0]

    # Initialization of the jump times list
    times = [0.0] * (len(idx_2) + len(idx_1))

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
            times[k:k+n_jumps] = np.random.uniform(t_start, t_end, size=n_jumps)

            # Update times list index
            k += n_jumps

    # Intervals with a single jump
    if len(idx_1) > 0:
        times[k:] = idx_1 * stepsize - 0.5 * stepsize

    # Lists concatenation and jump times sorted
    jump_times = np.concatenate([times])
    jump_times.sort()

    return jump_times
