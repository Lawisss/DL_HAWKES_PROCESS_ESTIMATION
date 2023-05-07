#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Evaluation module

File containing model evaluation functions

"""

from typing import Tuple, Union

import numpy as np
import pandas as pd
import polars as pl


# Absolute / Relative error function

def compute_errors(y_test: Union[np.ndarray, pl.DataFrame, pd.DataFrame], y_pred: Union[np.ndarray, pl.DataFrame, pd.DataFrame], model_name: str = "Benchmark") -> Tuple[np.ndarray, np.ndarray]:
    
    """
    Computed absolute error and relative error between true and predicted values

    Args:
        y_test (Union[np.ndarray, pl.DataFrame, pd.DataFrame]): True values
        y_pred (Union[np.ndarray, pl.DataFrame, pd.DataFrame]): Predicted values
        model_name (str, optional): Model name (default: "Benchmark")

    Returns:
        Absolute error and relative error
    """

    # Checked types
    y_test = y_test.to_numpy() if not isinstance(y_test, np.ndarray) else y_test
    y_pred = y_pred.to_numpy() if not isinstance(y_pred, np.ndarray) else y_pred

    # Computed absolute/relative error
    abs_error = np.abs(y_pred - y_test)
    relative_error = abs_error / y_test

    # Printed MAE and MRE
    print(f'{model_name} - MAE: {np.mean(abs_error):.4f}, MRE: {np.mean(relative_error):.4f}')

    return abs_error, relative_error
