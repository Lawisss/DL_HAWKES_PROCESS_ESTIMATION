#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Linear model module

File containing Linear Aggregated/Binned Hawkes Process estimation (alpha/beta)

"""

import os
from typing import Tuple, Optional, Callable

import numpy as np

import VARIABLES.evaluation_var as eval
from UTILS.utils import write_parquet

# Linear Regression function (alpha/beta estimation)

def linear_model(val_y_pred: np.ndarray, train_x: np.ndarray, val_x: np.ndarray, params: np.ndarray, step_size: float = 0.05, folder: str = "TESTING", args: Optional[Callable] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    """
    Calculates predicted alpha/beta values using linear regression

    Args: 
        val_y_pred (np.ndarray): Eta/mu predictions
        train_x (np.ndarray): Training data
        val_x (np.ndarray): Validation data
        params (np.ndarray): Parameter values
        step_size (float, optional): Step size for alpha values. (default: 0.05)
        folder (str, optional): Sub-folder name in RUNS folder (default: 'TESTING')
        args (Callable, optional): Arguments if you use main.py instead of tutorial.ipynb

    Returns:
        dict: Predicted alpha/beta values
    """

    # Arrays conversion
    val_y_pred = val_y_pred.numpy() if not isinstance(val_y_pred, np.ndarray) else val_y_pred
    train_x = train_x.numpy() if not isinstance(train_x, np.ndarray) else train_x
    val_x = val_x.numpy() if not isinstance(val_x, np.ndarray) else val_x

    # Predicted validation set values
    val_eta_pred = val_y_pred[:, 0]
    val_mu_pred = val_y_pred[:, 1]

    # Defined min and max eta values for comparison
    min_eta = val_eta_pred - 0.05
    max_eta = val_eta_pred + 0.05
    eta = params[:, 0] / params[:, 1]

    #Extracted similar eta values from training data using mask
    similar_eta = train_x[(eta > min_eta) & (eta < max_eta), :]
    similar_eta_alpha = params[(eta > min_eta) & (eta < max_eta), 0]
    _ = params[(eta > min_eta) & (eta < max_eta), 2]

    # Calculated max alpha and created step values
    max_alpha = int(np.ceil(np.max(params[:, 0])))
    min_v = np.arange(0, max_alpha, step_size, dtype=np.float32)
    max_v = np.arange(step_size, (max_alpha + step_size), step_size, dtype=np.float32)

    # Created mask and calculated statistics
    mask = np.logical_or.reduce([(similar_eta_alpha >= min_v) & (similar_eta_alpha < max_v) for min_v, max_v in zip(min_v, max_v)])
    stats = np.mean(np.max(similar_eta[mask], axis=1), axis=0)

    # Created array for alpha values
    x = np.arange((step_size / 2), (max_alpha * step_size), step_size, dtype=np.float32)[:len(stats)]
    a = np.vstack([x, np.ones(len(x), dtype=np.float32)]).T

    # Calculated coefficients
    coefs = np.dot(np.linalg.inv(np.dot(a.T, a)), np.dot(a.T, stats))
    slope, intercept = coefs[0], coefs[1]

    # Calculated predicted alpha/beta values using linear regression
    stat = np.median(np.max(val_x, axis=1))
    alpha_pred, beta_pred = slope * stat + intercept, alpha_pred / val_eta_pred

    # Default parameters
    default_params = {"logdirun": eval.LOGDIRUN, "run_name": eval.RUN_NAME}
    # Initialized parameters
    dict_args = {k: getattr(args, k, v) for k, v in default_params.items()}

    # Written parameters to Parquet file
    write_parquet({"alpha_pred": alpha_pred, "beta_pred": beta_pred, "val_eta_pred": val_eta_pred, "val_mu_pred": val_mu_pred}, 
                  filename=f"{dict_args['run_name']}_PRED.parquet", folder=os.path.join(dict_args['logdirun'], folder, dict_args['run_name']))

    return np.array([alpha_pred, beta_pred, val_eta_pred, val_mu_pred], dtype=np.float32).T, alpha_pred, beta_pred 
