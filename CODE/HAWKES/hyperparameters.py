#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Hyper-parameters module

File containing Hawkes process hyper-parameters generation functions (Default Parameter Values)

"""

from typing import Tuple

import numpy as np

import VARIABLES.hawkes_var as hwk
from UTILS.utils import write_parquet



# Generated Hawkes process hyper-parameters (alpha, beta, mu)

def hyper_params_simulation(filename: str = "hawkes_hyperparams.parquet") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

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
    
    # Generated random vectors of size PROCESS_NUM (epsilon = average of events)
    epsilon = np.random.normal(hwk.EXPECTED_ACTIVITY, hwk.STD, hwk.PROCESS_NUM)
    eta = np.random.uniform(hwk.MIN_ITV_ETA, hwk.MAX_ITV_ETA, hwk.PROCESS_NUM)
    beta = np.random.uniform(hwk.MIN_ITV_BETA, hwk.MAX_ITV_BETA, hwk.PROCESS_NUM)

    # Calculated alpha/mu vectors from beta/eta vectors (alpha = eta because of library exponential formula)
    alpha = eta
    mu = (epsilon / hwk.TIME_HORIZON) * (1 - eta)

    # Written parameters to Parquet file
    write_parquet({"alpha": alpha, "beta": beta, "mu": mu}, filename=filename)
    
    # Created dictionaries list containing the parameters
    # params = list(map(lambda a, b, m: {"alpha": a, "beta": b, "mu": m}, alpha, beta, mu)) 

    # Written parameters to CSV file 
    # write_csv(params, filename=filename) 

    return np.array([alpha, beta, mu], dtype=np.float32).T, alpha, beta, mu


