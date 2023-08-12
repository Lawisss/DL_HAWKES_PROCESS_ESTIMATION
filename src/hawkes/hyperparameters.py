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


# Generated Hawkes process exponential hyperparameters (alpha, beta, mu)

def exp_hyperparams(record: bool = True, filename: Optional[str] = "exp_hawkes_hyperparams.parquet", args: Optional[Callable] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    """
    Generated and saved Hawkes process exponential hyperparameters

    Args:
        record (bool, optional): Record results in parquet file (default: True)
        filename (str, optional): Filename to save hyperparameters in parquet file (default: "exp_hawkes_hyperparams.parquet")
        args (Callable, optional): Arguments if you use run.py instead of tutorial.ipynb

    Returns:
        A tuple containing:
        - Alpha, beta, eta and mu parameters for each process
        - Alpha parameters for each process
        - Beta parameters for each process
        - Eta parameters for each process
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

    # Written parquet file
    if record is True:
        write_parquet(pl.DataFrame({"alpha": alpha, "beta": beta, "eta": eta, "mu": mu}), filename=filename)

    return np.array([alpha, beta, eta, mu], dtype=np.float32).T, alpha, beta, eta, mu


# Generated Hawkes process power law hyperparameters (k, c, p)

def pow_hyperparams(record: bool = True, filename: Optional[str] = "pow_hawkes_hyperparams.parquet", args: Optional[Callable] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    """
    Generated and saved Hawkes process power law hyperparameters

    Args:
        record (bool, optional): Record results in parquet file (default: True)
        filename (str, optional): Filename to save hyperparameters in parquet file (default: "pow_hawkes_hyperparams.parquet")
        args (Callable, optional): Arguments if you use run.py instead of tutorial.ipynb

    Returns:
        A tuple containing:
        - k, c, p, eta and mu parameters for each process
        - k parameters for each process
        - c parameters for each process
        - p parameters for each process
        - Eta parameters for each process
        - Mu parameters for each process
    """

    # Default parameters
    default_params = {"expected_activity": hwk.EXPECTED_ACTIVITY,
                      "std": hwk.STD,
                      "process_num": hwk.PROCESS_NUM,
                      "min_itv_eta": hwk.MIN_ITV_ETA,
                      "max_itv_eta": hwk.MAX_ITV_ETA,
                      "min_itv_k": hwk.MIN_ITV_K,
                      "max_itv_k": hwk.MAX_ITV_K,
                      "min_itv_c": hwk.MIN_ITV_C,
                      "max_itv_c": hwk.MAX_ITV_C,
                      "min_itv_p": hwk.MIN_ITV_P,
                      "max_itv_p": hwk.MAX_ITV_P,
                      "time_horizon": hwk.TIME_HORIZON}

    # Initialized parameters
    dict_args = {k: getattr(args, k, v) for k, v in default_params.items()}

    # Generated random vectors for power-law hyperparameters
    k = np.random.exponential(scale=(dict_args['min_itv_k'] + dict_args['max_itv_k']) / 2, size=dict_args['process_num'])
    c = np.random.exponential(scale=(dict_args['min_itv_c'] + dict_args['max_itv_c']) / 2, size=dict_args['process_num'])
    p = np.random.uniform(dict_args['min_itv_p'], dict_args['max_itv_p'], dict_args['process_num'])

    # Generated random vectors for hawkes hyperparameters
    epsilon = np.random.normal(dict_args['expected_activity'], dict_args['std'], dict_args['process_num'])
    eta = np.random.uniform(dict_args['min_itv_eta'], dict_args['max_itv_eta'], dict_args['process_num'])

    # Calculated mu vectors from eta vectors
    mu = (epsilon / dict_args['time_horizon']) * (1 - eta)

    # Written parquet file
    if record is True:
        write_parquet(pl.DataFrame({"k": k, "c": c, "p": p, "eta": eta, "mu": mu}), filename=filename)

    return np.array([k, c, p, eta, mu], dtype=np.float32).T, k, c, p, eta, mu

