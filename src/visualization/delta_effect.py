#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Discretization step effect module

File containing discretization step effect function

"""

from typing import List, Callable, Optional

import numpy as np
import polars as pl

import variables.hawkes_var as hwk
from tools.utils import write_parquet
from hawkes.simulation import hawkes_simulation


# Deltas simulations function

def delta_simulations(deltas: List = [0.25, 0.5, 1.0, 2.0, 5.0], args: Optional[Callable] = None) -> np.ndarray:

    """
    Discretized jump times into binned histogram for each deltas, where bin are time interval of length "hwk.DISCRETISE_STEP"

    Args:
        deltas (List, optional): Discretisation step to test (default: [0.25, 0.5, 1.0, 2.0, 5.0])
        args (Callable, optional): Arguments if you use run.py instead of tutorial.ipynb

    Returns:
        np.ndarray: Binned histogram counts for each process of each delta, where "num_bins" is number of bins used to discretize jump times
    """  

    # Default parameters
    default_params = {"expected_activity": hwk.EXPECTED_ACTIVITY,
                      "std": hwk.STD,
                      "process_num": hwk.PROCESS_NUM,
                      "min_itv_eta": hwk.MIN_ITV_ETA,
                      "max_itv_eta": hwk.MAX_ITV_ETA,
                      "min_itv_beta": hwk.MIN_ITV_BETA,
                      "max_itv_beta": hwk.MAX_ITV_BETA,
                      "time_horizon": hwk.TIME_HORIZON,
                      "test_num": hwk.TEST_NUM}

    # Initialized parameters
    dict_args = {k: getattr(args, k, v) for k, v in default_params.items()}
    counts = []

    for delta in deltas:

        # Generated random vectors of size PROCESS_NUM (epsilon = average of events)
        epsilon = np.random.normal(dict_args['expected_activity'], dict_args['std'], dict_args['test_num'])
        eta = np.random.uniform(dict_args['min_itv_eta'], dict_args['max_itv_eta'], dict_args['test_num'])
        beta = np.random.uniform(dict_args['min_itv_beta'], dict_args['max_itv_beta'], dict_args['test_num'])

        # Calculated alpha/mu vectors from beta/eta vectors (alpha = eta because of library exponential formula)
        alpha = eta
        mu = (epsilon / dict_args['time_horizon']) * (1 - eta)

        # Computed bins number
        num_bins = int(dict_args['time_horizon'] // delta)
        
        # Hawkes processes for 100 tests
        simulated_events_seqs = [np.array(hawkes_simulation(params={"mu": mu[i], "alpha": alpha[i], "beta": beta[i]})[1], dtype=np.float32) for _ in range(dict_args["process_num"]) for i in range(dict_args['test_num'])]
        
        # Initialized array with dimensions (number of processes, number of jumps per unit of time)
        count = np.zeros((len(simulated_events_seqs), num_bins), dtype=np.float32)

        # For each process (j), compute jump times histogram (h) using intervals boundaries specified by bins
        for j, h in enumerate(simulated_events_seqs):
            count[j], _ = np.histogram(h, bins=np.linspace(0, dict_args['time_horizon'], num_bins + 1))

        # Written parquet file
        counts.append(count)
        write_parquet(pl.DataFrame({"alpha": alpha, "beta": beta, "eta": eta, "mu": mu}), filename=f"hawkes_hyperparams_delta_{delta}.parquet")
        write_parquet(pl.DataFrame(count, schema=np.arange(num_bins, dtype=np.int32).astype(str).tolist()), filename=f"binned_hawkes_simulations_delta_{delta}.parquet")
        

    return counts

