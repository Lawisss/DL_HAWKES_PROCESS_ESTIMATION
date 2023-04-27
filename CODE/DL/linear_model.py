#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Linear model module

File containing Linear Aggregated/Binned Hawkes Process estimation (alpha/beta)

"""

import os
from typing import Tuple, Callable

import numpy as np

from DL.mlp_model import MLPTrainer
from UTILS.utils import write_parquet

# Linear Regression function (alpha/beta estimation)

def linear_model(train_x: np.ndarray, val_x: np.ndarray, params: np.ndarray, model: Callable = MLPTrainer(), step_size: float = 0.05, filename: str = "alpha_beta_linear_estimation.parquet") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    """
    Calculates predicted alpha/beta values using linear regression

    Args: 
        train_x (np.ndarray): Training data
        val_x (np.ndarray): Validation data
        params (np.ndarray): Parameter values
        model (Callable, optional): Trained model
        step_size (float, optional): Step size for alpha values. (default: 0.05)
        filename (str, optional): Filename to save parameters in Parquet file (default: "alpha_beta_linear_estimation.parquet")

    Returns:
        dict: Predicted alpha/beta values, validation set mu and eta median
    """

    # Predicted validation set values
    _, val_eta, val_mu = model.predict(val_x)

    # Defined min and max eta values for comparison
    min_eta = val_eta - 0.05
    max_eta = val_eta + 0.05
    eta = params[:, 0] / params[:, 1]

    #Extracted similar eta values from training data using mask
    similar_eta = train_x[(eta > min_eta) & (eta < max_eta), :]
    similar_eta_alpha = params[(eta > min_eta) & (eta < max_eta), 0]
    similar_eta_mu = params[(eta > min_eta) & (eta < max_eta), 2]

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
    alpha_pred, beta_pred = slope * stat + intercept, alpha_pred / val_eta

    # Written parameters to Parquet file
    write_parquet({"alpha_pred": alpha_pred, "beta_pred": beta_pred, "val_mu": val_mu, "val_eta": val_eta}, filename=filename)

    return np.array([alpha_pred, beta_pred, val_mu, val_eta], dtype=np.float32).T, alpha_pred, beta_pred, val_mu, val_eta
