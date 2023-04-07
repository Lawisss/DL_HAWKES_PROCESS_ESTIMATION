#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Hyper-parameters module

File containing Hawkes process hyper-parameters generation functions (Default Parameter Values).

"""

import numpy as np
import pandas as pd

import VARIABLES.variables as var

# Generated Hawkes process hyper-parameters (alpha, beta, mu)

def hyper_params_simulation(filename="hawkes_hyperparams.csv"):

    # Generated random vectors of size PROCESS_NUM
    epsilon = np.random.normal(var.EXPECTED_ACTIVITY, var.STD, var.PROCESS_NUM)
    eta = np.random.uniform(var.MIN_ITV_ETA, var.MAX_ITV_ETA, var.PROCESS_NUM)
    beta = np.random.uniform(var.MIN_ITV_BETA, var.MAX_ITV_BETA, var.PROCESS_NUM)

    # Calculated alpha and mu vectors from beta and eta vectors (alpha = eta because of library exponential formula)
    alpha = eta
    mu = (epsilon / var.TIME_HORIZON) * (1 - eta)

    # Created a DataFrame, name the columns, and generate csv file
    df = pd.DataFrame({"alpha": alpha, "beta": beta, "mu": mu})
    df.to_csv(f"{var.FILEPATH}{filename}", index=False)

    return np.array([alpha, beta, mu]).T, alpha, beta, mu