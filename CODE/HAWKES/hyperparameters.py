#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Hyper-parameters module

File containing Hawkes process hyper-parameters generation functions (Default Parameter Values).

"""

import numpy as np
from typing import Tuple

import VARIABLES.hawkes_var as hwk
from UTILS.utils import write_csv

# Generated Hawkes process hyper-parameters (alpha, beta, mu)

def hyper_params_simulation(filename: str = "hawkes_hyperparams.csv") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    # Generated random vectors of size PROCESS_NUM
    epsilon = np.random.normal(hwk.EXPECTED_ACTIVITY, hwk.STD, hwk.PROCESS_NUM)
    eta = np.random.uniform(hwk.MIN_ITV_ETA, hwk.MAX_ITV_ETA, hwk.PROCESS_NUM)
    beta = np.random.uniform(hwk.MIN_ITV_BETA, hwk.MAX_ITV_BETA, hwk.PROCESS_NUM)

    # Calculated alpha/mu vectors from beta/eta vectors (alpha = eta because of library exponential formula)
    alpha = eta
    mu = (epsilon / hwk.TIME_HORIZON) * (1 - eta)

    # Created dictionaries list containing the parameters
    params = list(map(lambda a, b, m: {"alpha": a, "beta": b, "mu": m}, alpha, beta, mu)) 

    # Written parameters to a CSV file 
    write_csv(params, filename=filename) 

    return np.array([alpha, beta, mu], dtype=np.float32).T, alpha, beta, mu


