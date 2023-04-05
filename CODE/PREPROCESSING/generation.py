#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Generation module

File containing Training/Testing dataset generation functions (Default Parameter Values).

"""

import numpy as np
import pandas as pd

import VARIABLES.variables as var

# Generated training/testing dataset

def data_generation(filepath="C:/Users/Nicolas Girard/Documents/VAE_HAWKES_PROCESS_ESTIMATION/CODE/RESULTS/hawkes_params.csv"):

    # Generated random vectors of size TRAINING_PROCESS
    epsilon = np.random.normal(var.EXPECTED_ACTIVITY, var.STD, var.TRAINING_PROCESS)
    eta = np.random.uniform(var.MIN_ITV_ETA, var.MAX_ITV_ETA, var.TRAINING_PROCESS)
    beta = np.random.uniform(var.MIN_ITV_BETA, var.MAX_ITV_BETA, var.TRAINING_PROCESS)

    # Calculated alpha and mu vectors from beta and eta vectors
    alpha = beta * eta
    mu = (epsilon / var.TIME_HORIZON) * (1 - eta)

    # Stacked params to create a matrix of size (TRAINING_PROCESS, 3)
    params = np.column_stack((alpha, beta, mu))

    # Created a DataFrame, name the columns, and generate csv file
    df = pd.DataFrame(params, columns=["alpha", "beta", "mu"])
    df.to_csv(filepath, index=False)

    return params, alpha, beta, mu