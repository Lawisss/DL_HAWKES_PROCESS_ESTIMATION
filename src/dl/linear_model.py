#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Linear model module

File containing Linear Aggregated/Binned Hawkes Process estimation (alpha/beta)

"""

import os
from typing import Tuple, Optional, Callable

import numpy as np
import polars as pl

from dl.mlp_model import MLPTrainer
import variables.eval_var as eval
from tools.utils import write_parquet

# Linear Regression function (alpha/beta estimation)

def linear_model(train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, step_size: float = 0.05, args: Optional[Callable] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    """
    Calculates predicted alpha/beta values using linear regression

    Args: 
        train_x (np.ndarray): Inputs training data
        train_y (np.ndarray): Labels training data
        val_x (np.ndarray): Inputs validation data
        step_size (float, optional): Step size for alpha values (default: 0.05)
        args (Callable, optional): Arguments if you use run.py instead of tutorial.ipynb (default: None)

    Returns:
        dict: Predicted alpha/beta values
    """

    # Loaded model and predictions
    model = MLPTrainer()
    print(model.load_model())
    val_y_pred, val_eta_pred, val_mu_pred = model.predict(val_x)

    # Arrays conversion
    val_y_pred, val_x, train_x, train_y = map(lambda x: x.numpy().astype(np.float32), (val_y_pred, val_x, train_x, train_y))
    val_eta_pred, val_mu_pred = map(lambda x: np.array([x], dtype=np.float32), (val_eta_pred, val_mu_pred))

    # Defined min and max eta values for comparison
    min_eta = val_eta_pred - step_size
    max_eta = val_eta_pred + step_size

    # Kernel in hawkes librairie (alpha = eta)
    eta = train_y[:, 0] # train_y[:, 0] / train_y[:, 1]

    # Extracted similar eta values from training data using mask
    similar_eta_processes = train_x[(eta > min_eta) & (eta < max_eta), :]
    similar_eta_alpha = train_y[(eta > min_eta) & (eta < max_eta), 0]

    # Calculated max alpha and created step values
    max_alpha = int(np.ceil(np.max(train_y[:, 0])))
    stat_data = np.empty((max_alpha*20), dtype=np.float32)
   
    # Computed intervals
    min_v = np.arange(0, max_alpha - step_size, step_size)
    max_v = np.arange(step_size, max_alpha, step_size)

    # Computed statistics
    for i in range(max_alpha*20):
        mask = (similar_eta_alpha >= min_v[i-1]) & (similar_eta_alpha < max_v[i-1])
        stat_data[i-1] = np.mean(np.max(similar_eta_processes[mask], axis=1), dtype=np.float32) if np.any(mask) else np.nan
        
    # Linear regression preparation
    x = np.vstack([stat_data[~np.isnan(stat_data)], np.ones(len(stat_data[~np.isnan(stat_data)]))]).T
    y = np.arange(step_size / 2, max_alpha * 20 * step_size, step_size)[~np.isnan(stat_data)]

    # Applied linear algebra formula about linear regression
    coefs = np.dot(np.linalg.inv(np.dot(x.T, x)), np.dot(x.T, y.reshape(-1, 1)))
    slope, intercept = coefs

    # Predicted and printed results
    stat = np.median(np.max(val_x, axis=1))
    alpha_pred = (slope * stat) + intercept
    beta_pred = 1 / val_eta_pred # alpha_pred / val_eta_pred
    print(f"Linear Regression (Slope: {slope[0]:.4f}, Intercept: {intercept[0]:.4f}) - Estimated self-exciting rate (α): {alpha_pred[0]:.4f}, Estimated decay rate (β): {beta_pred[0]:.4f}")


    # Default parameters
    default_params = {"logdirun": eval.LOGDIRUN, "test_dir": eval.TEST_DIR, "run_name": eval.RUN_NAME}

    # Initialized parameters
    dict_args = {k: getattr(args, k, v) for k, v in default_params.items()}

    # Written parameters to parquet file
    write_parquet(pl.DataFrame({"alpha_pred_avg": alpha_pred, "beta_pred_avg": beta_pred, "val_eta_pred": val_eta_pred, "val_mu_pred": val_mu_pred}), 
                  filename=f"{dict_args['run_name']}_linear_predictions.parquet", folder=os.path.join(dict_args['logdirun'], dict_args['test_dir'], dict_args['run_name']))

    return np.array([alpha_pred, beta_pred, val_eta_pred, val_mu_pred], dtype=np.float32).T, alpha_pred, beta_pred 
