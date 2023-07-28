#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Evaluation module

File containing model evaluation functions

"""

from typing import Optional, Union, Callable

import numpy as np
import pandas as pd
import polars as pl
from scipy.integrate import quad

import variables.hawkes_var as hwk
from tools.utils import write_parquet
from hawkes.simulation import hawkes_intensity


# Error / Relative error function

def compute_errors(y_test: Union[np.ndarray, pl.DataFrame, pd.DataFrame], y_pred: Union[np.ndarray, pl.DataFrame, pd.DataFrame], model_name: Optional[str] = "Benchmark", record: bool = True, filename: Optional[str] = "errors.parquet") -> pl.DataFrame:
    
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


# Intensity integration function

def integrate_intensity(params: Union[np.ndarray, pl.DataFrame, pd.DataFrame], simulated_events_seqs: Union[np.ndarray, pl.DataFrame, pd.DataFrame], decoded_intensity: Union[np.ndarray, pl.DataFrame, pd.DataFrame], record: bool = True, filename: Optional[str] = "intensity_integration.parquet", args: Optional[Callable] = None) -> pl.DataFrame:

    """
    Computed integrated intensities to compare with decoded intensities

    Args:
        params (Union[np.ndarray, pl.DataFrame, pd.DataFrame]) Hawkes process hyperparameters
        simulated_events_seqs (Union[np.ndarray, pl.DataFrame, pd.DataFrame]): Simulated event sequences of each Hawkes process
        decoded_intensity (Union[np.ndarray, pl.DataFrame, pd.DataFrame]): Predicted intensities
        record (bool, optional): Record results in parquet file (default: True)
        filename (str, optional): Parquet filename to save results (default: "intensity_error.parquet")
        args (Callable, optional): Arguments if you use run.py instead of tutorial.ipynb (default: None)

    Returns:
        pl.DataFrame: Decoded intensity and Integrated intensity
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


# NRMSE error function

def compute_nrmse(params: Union[np.ndarray, pl.DataFrame, pd.DataFrame], simulated_events_seqs: Union[np.ndarray, pl.DataFrame, pd.DataFrame], decoded_intensity: Union[np.ndarray, pl.DataFrame, pd.DataFrame], record: bool = True, filename: Optional[str] = "nrmse_errors.parquet") -> pl.DataFrame:
    
    """
    Computed NRMSE error

    Args:
        params (Union[np.ndarray, pl.DataFrame, pd.DataFrame]): Hawkes processes hyperparameters
        simulated_events_seqs (Union[np.ndarray, pl.DataFrame, pd.DataFrame]): Hawkes processes
        decoded_intensities (Union[np.ndarray, pl.DataFrame, pd.DataFrame]): Predicted intensities
        record (bool, optional): Record results in parquet file (default: True)
        filename (str, optional): Parquet filename to save results (default: "nrmse_errors.parquet")
        args (Callable, optional): Arguments if you use run.py instead of tutorial.ipynb (default: None)

    Returns:
        pl.DataFrame: NRMSE error
    """

    # Checked types
    params = params.to_numpy() if not isinstance(params, np.ndarray) else params
    simulated_events_seqs = simulated_events_seqs.to_numpy() if not isinstance(simulated_events_seqs, np.ndarray) else simulated_events_seqs
    decoded_intensity = decoded_intensity.to_numpy() if not isinstance(decoded_intensity, np.ndarray) else decoded_intensity
    
    # Initialized parameters
    _, beta, _ = params[0, 0], params[0, 1], params[0, 3]
    t = (np.arange(0, (100 / 0.001), dtype=np.float32) * 0.001)
    intensity = np.zeros((100, len(t)), dtype=np.float32)
    integrated_intensity = np.zeros((100, 100), dtype=np.float32)
    
    # Computed intensity
    for i in range(100):
        for j in range(len(t)):
            intensity[i, j] = np.sum(np.exp(beta * (simulated_events_seqs[i, 0][simulated_events_seqs[i, 0] < t[j]] - t[j])), dtype=np.float32)
    
    # Computed integrated intensity
    for i in range(100):
        for k in range(1, 100 + 1):
            integrated_intensity[k-1] = quad(lambda x: intensity[i, int(x * 1000)], k-1, k, limit=1000, epsabs=0.01)[0]
    
    # Computed NRMSE
    nrmse = [np.sqrt(np.mean((decoded_intensities - integrated_intensities)**2)) / (np.max(integrated_intensities) - np.min(integrated_intensities)) for decoded_intensities, integrated_intensities in zip(integrated_intensity.tolist(), decoded_intensity.tolist())]

    # Written parquet file
    if record is True:
        write_parquet(pl.DataFrame(np.column_stack((nrmse)), schema=["nrmse_error"]), filename=filename)

    return pl.DataFrame(np.column_stack((nrmse)), schema=["nrmse_error"])