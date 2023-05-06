#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Hyper-parameters module

File containing Hawkes process hyper-parameters generation functions (Default Parameter Values)

"""

from typing import Tuple, Optional, Callable

import numpy as np
import polars as pl

import variables.hawkes_var as hwk
from tools.utils import write_parquet



# Generated Hawkes process hyper-parameters (alpha, beta, mu)

def hyper_params_simulation(filename: str = "hawkes_hyperparams.parquet", args: Optional[Callable] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    """
    Generated and saved Hawkes process hyperparameters

    Args:
        filename (str, optional): Filename to save hyperparameters in parquet file (default: "hawkes_hyperparams.parquet")
        args (Callable, optional): Arguments if you use run.py instead of tutorial.ipynb

    Returns:
        A tuple containing:
        - Alpha, beta, and mu parameters for each process
        - Alpha parameters for each process
        - Beta parameters for each process
        - Mu parameters for each process
    """

    # Default parameters
    default_params = {"expected_activity": hwk.EXPECTED_ACTIVITY,
                      "std": hwk.STD,
                      "process_num": hwk.PROCESS_NUM,
                      "min_itv_eta": hwk.MIN_ITV_ETA,
                      "max_itv_eta": hwk.MAX_ITV_ETA,
                      "min_itv_beta": hwk.MIN_ITV_BETA,
                      "max_itv_beta": hwk.MAX_ITV_BETA,
                      "time_horizon": hwk.TIME_HORIZON}

    # Initialized parameters
    dict_args = {k: getattr(args, k, v) for k, v in default_params.items()}

    # Generated random vectors of size PROCESS_NUM (epsilon = average of events)
    epsilon = np.random.normal(dict_args['expected_activity'], dict_args['std'], dict_args['process_num'])
    eta = np.random.uniform(dict_args['min_itv_eta'], dict_args['max_itv_eta'], dict_args['process_num'])
    beta = np.random.uniform(dict_args['min_itv_beta'], dict_args['max_itv_beta'], dict_args['process_num'])

    # Calculated alpha/mu vectors from beta/eta vectors (alpha = eta because of library exponential formula)
    alpha = eta
    mu = (epsilon / dict_args['time_horizon']) * (1 - eta)

    # Written parameters to parquet file
    write_parquet(pl.DataFrame({"alpha": alpha, "beta": beta, "mu": mu}).with_columns(pl.col(pl.Float64).cast(pl.Float32)), filename=filename)
    
    # Created dictionaries list containing the parameters
    # params = list(map(lambda a, b, m: {"alpha": a, "beta": b, "mu": m}, alpha, beta, mu)) 

    # Written parameters to CSV file 
    # write_csv(params, filename=filename) 

    return np.array([alpha, beta, mu], dtype=np.float32).T, alpha, beta, mu


