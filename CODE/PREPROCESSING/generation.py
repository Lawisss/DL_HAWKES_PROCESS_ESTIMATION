#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Generation module

File containing Training/Testing dataset generation functions (Default Parameter Values).

"""

import numpy as np
import pandas as pd

import VARIABLES.variables as var


def data_generation(filepath="C:/Users/Nicolas Girard/Documents/VAE_HAWKES_PROCESS_ESTIMATION/CODE/RESULTS/hawkes_params.csv"):

    epsilon = np.random.normal(var.EXPECTED_ACTIVITY, var.STD, var.TRAINING_PROCESS)
    eta = np.random.uniform(var.MIN_ITV_ETA, var.MAX_ITV_ETA, var.TRAINING_PROCESS)
    beta = np.random.uniform(var.MIN_ITV_BETA, var.MAX_ITV_BETA, var.TRAINING_PROCESS)

    alpha = beta * eta
    mu = (epsilon / var.TIME_HORIZON) * (1 - eta)
    
    params = np.array([alpha, beta, mu]).reshape(var.TRAINING_PROCESS, 3)
    df = pd.DataFrame({"alpha": alpha, "beta": beta, "mu": mu})
    df.to_csv(filepath)

    return params, alpha, beta, mu