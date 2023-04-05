#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Discretisation module

File containing Aggregated Hawkes Process functions (Hawkes Process conversion).

"""

import numpy as np

def discretise(jump_times, delta, horizon):

    k = int(np.floor(horizon / delta))
    n = len(jump_times)
    counts = np.zeros((n, k))

    for j in range(n):
        h = jump_times[j]
        l = len(h)

        for i in range(l):
            w = int(np.ceil(h[i] / delta))
            counts[j, w-1] += 1

    return counts

def temp_func(jump_times, horizon):

    if len(jump_times) == 0:
        stepsize = horizon
    else:
        times = np.concatenate(([0], jump_times, [horizon]))
        diff = times[1:] - times[:-1]
        stepsize = np.around(np.min(diff[np.nonzero(diff)]), 1)

    return stepsize

def find_stepsize(jump_times, horizon):
    return np.min([temp_func(x, horizon) for x in jump_times])

def jump_times(h, horizon):

    stepsize = horizon / len(h)
    times = []
    idx_1 = np.where(h == 1)[0]
    idx_2 = np.where(h > 1)[0]

    if len(idx_2) > 0:
        for i in range(len(idx_2)):
            times += list(np.random.uniform((idx_2[i] - 1) * stepsize, idx_2[i] * stepsize, size=h[idx_2[i]]))

    times += list((idx_1 * stepsize)-(.5 * stepsize))
    jump_times = np.sort(times)

    return jump_times
