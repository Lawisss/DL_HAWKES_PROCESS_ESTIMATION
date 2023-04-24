#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Hyper-parameters module

File containing Hawkes process hyper-parameters generation functions (Default Parameter Values)

"""

from typing import Tuple

import numpy as np

import VARIABLES.hawkes_var as hwk
from UTILS.utils import argparser, write_parquet



# Generated Hawkes process hyper-parameters (alpha, beta, mu)

@argparser(parse_args=False, arg_groups=['hawkes_params', 'hawkes_simulation_params'])
def hyper_params_simulation(args_parsed, filename: str = "hawkes_hyperparams.parquet") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    """
    Generated and saved Hawkes process hyperparameters

    Args:
        filename (str, optional): Filename to save hyperparameters in Parquet file (default: "hawkes_hyperparams.parquet")

    Returns:
        A tuple containing:
        - Alpha, beta, and mu parameters for each process
        - Alpha parameters for each process
        - Beta parameters for each process
        - Mu parameters for each process
    """

    # Parameters initialization
    expected_activity, std, process_num, min_itv_eta, max_itv_eta, min_itv_beta, max_itv_beta, time_horizon = \
    (args_parsed.expected_activity, args_parsed.std, args_parsed.process_num, args_parsed.min_itv_eta, args_parsed.max_itv_eta, args_parsed.min_itv_beta, args_parsed.max_itv_beta, args_parsed.time_horizon) \
    if args_parsed else (hwk.EXPECTED_ACTIVITY, hwk.STD, hwk.PROCESS_NUM, hwk.MIN_ITV_ETA, hwk.MAX_ITV_ETA, hwk.MIN_ITV_BETA, hwk.MAX_ITV_BETA, hwk.TIME_HORIZON)
    
    # Generated random vectors of size PROCESS_NUM (epsilon = average of events)
    epsilon = np.random.normal(expected_activity, std, process_num)
    eta = np.random.uniform(min_itv_eta, max_itv_eta, process_num)
    beta = np.random.uniform(min_itv_beta, max_itv_beta, process_num)

    # Calculated alpha/mu vectors from beta/eta vectors (alpha = eta because of library exponential formula)
    alpha = eta
    mu = (epsilon / time_horizon) * (1 - eta)

    # Written parameters to Parquet file
    write_parquet({"alpha": alpha, "beta": beta, "mu": mu}, filename=filename)
    
    # Created dictionaries list containing the parameters
    # params = list(map(lambda a, b, m: {"alpha": a, "beta": b, "mu": m}, alpha, beta, mu)) 

    # Written parameters to CSV file 
    # write_csv(params, filename=filename) 

    return np.array([alpha, beta, mu], dtype=np.float32).T, alpha, beta, mu


