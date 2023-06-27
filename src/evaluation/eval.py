#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Evaluation module

File containing model evaluation functions

"""

from typing import Optional, Tuple, Union, Callable

import numpy as np
import pandas as pd
import polars as pl
from scipy.integrate import quad

import variables.hawkes_var as hwk
from tools.utils import write_parquet
from hawkes.simulation import hawkes_intensity


# Error / Relative error function

def compute_errors(y_test: Union[np.ndarray, pl.DataFrame, pd.DataFrame], y_pred: Union[np.ndarray, pl.DataFrame, pd.DataFrame], model_name: Optional[str] = "Benchmark", record: bool = True, filename: Optional[str] = "errors.parquet") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    """
    Computed absolute error and relative error between true and predicted values

    Args:
        y_test (Union[np.ndarray, pl.DataFrame, pd.DataFrame]): True values
        y_pred (Union[np.ndarray, pl.DataFrame, pd.DataFrame]): Predicted values
        model_name (str, optional): Model name (default: "Benchmark")
        record (bool, optional): Record results in parquet file (default: True)
        filename (str, optional): Parquet filename to save results (default: "errors.parquet")

    Returns:
        Branching ratio / Baseline intensity error and relative error
    """

    # Checked types
    y_test = y_test.to_numpy() if not isinstance(y_test, np.ndarray) else y_test
    y_pred = y_pred.to_numpy() if not isinstance(y_pred, np.ndarray) else y_pred

    # Computed absolute/relative error
    eta_error = y_pred[:, 0] - y_test[:, 0]
    mu_error = y_pred[:, 1] - y_test[:, 1]

    eta_rel_error = eta_error / y_test[:, 0]
    mu_rel_error = mu_error / y_test[:, 1]

    # Printed MAE and MRE
    print(pl.DataFrame({"Model": model_name,
                        "Error Average (η)": round(np.mean(eta_error), 4),
                        "Error Average (μ)": round(np.mean(mu_error), 4),
                        "MRE (η)": round(np.mean(eta_rel_error), 4),
                        "MRE (μ)": round(np.mean(mu_rel_error), 4)}).with_columns(pl.col(pl.Float64).cast(pl.Float32)))

    # Written parquet file
    if record is True:
        write_parquet(pl.DataFrame(np.column_stack((eta_error, eta_rel_error, mu_error, mu_rel_error)), schema=["eta_error", "eta_rel_error", "mu_error", "mu_rel_error"]), filename=filename)

    return pl.DataFrame(np.column_stack((eta_error, eta_rel_error, mu_error, mu_rel_error)), schema=["eta_error", "eta_rel_error", "mu_error", "mu_rel_error"])


# Normalised Root Mean Square Error (NRMSE) function

def compute_nrmse(params: Union[np.ndarray, pl.DataFrame, pd.DataFrame], simulated_events_seqs: Union[np.ndarray, pl.DataFrame, pd.DataFrame], decoded_intensity: Union[np.ndarray, pl.DataFrame, pd.DataFrame], record: bool = True, filename: Optional[str] = "intensity_error.parquet", args: Optional[Callable] = None) -> Tuple[np.ndarray, np.ndarray]:

    """
    Computed NRMSE between predicted intensities and the integrated intensities.

    Args:
        params (Union[np.ndarray, pl.DataFrame, pd.DataFrame]) Hawkes process hyperparameters
        simulated_events_seqs (Union[np.ndarray, pl.DataFrame, pd.DataFrame]): Simulated event sequences of each Hawkes process
        decoded_intensity (Union[np.ndarray, pl.DataFrame, pd.DataFrame]): Predicted intensities
        record (bool, optional): Record results in parquet file (default: True)
        filename (str, optional): Parquet filename to save results (default: "intensity_error.parquet")
        args (Callable, optional): Arguments if you use run.py instead of tutorial.ipynb (default: None)

    Returns:
        pl.DataFrame: NRMSE
    """
        
    # Default parameters
    default_params = {"time_horizon": hwk.TIME_HORIZON}

    # Initialized parameters
    dict_args = {k: getattr(args, k, v) for k, v in default_params.items()}

    # Checked types
    params = params.to_numpy() if not isinstance(params, np.ndarray) else params
    simulated_events_seqs = simulated_events_seqs.to_numpy() if not isinstance(simulated_events_seqs, np.ndarray) else simulated_events_seqs
    decoded_intensity = decoded_intensity.to_numpy() if not isinstance(decoded_intensity, np.ndarray) else decoded_intensity

    # Initialized intensities
    true_intensity = hawkes_intensity(params, simulated_events_seqs)
    integrated_intensity = np.zeros(dict_args['time_horizon'], dtype=np.float32)

    # Computed integrated intensity
    for k in range(1, dict_args['time_horizon'] + 1):
        integrated_intensity[k-1] = quad(lambda x: true_intensity[int(x * 1000)], k-1, k, limit=1000, epsabs=0.01)[0]

    # Written parquet file
    if record is True:
        write_parquet(pl.DataFrame(np.column_stack((decoded_intensity[0], integrated_intensity)), schema=["decoded_intensity", "integrated_intensity"]), filename=filename)

    return pl.DataFrame(np.column_stack((decoded_intensity[0], integrated_intensity)), schema=["decoded_intensity", "integrated_intensity"])