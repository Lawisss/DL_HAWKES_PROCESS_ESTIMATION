#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Evaluation module

File containing model evaluation functions

"""

from typing import Tuple, Union

import numpy as np
import pandas as pd

def compute_errors(y_test: Union[np.ndarray, pd.DataFrame], y_pred: Union[np.ndarray, pd.DataFrame], model_name: str = 'Benchmark') -> Tuple[np.ndarray, np.ndarray]:
    
    """
    Computed absolute error and relative error between true and predicted values

    Args:
        y_test (Union[np.ndarray, pd.DataFrame]): True values
        y_pred (Union[np.ndarray, pd.DataFrame]): Predicted values
        model_name (str, optional): Model name

    Returns:
        Absolute error and relative error
    """

    y_test = y_test.values if not isinstance(y_test, np.ndarray) else y_test
    y_pred = y_pred.values if not isinstance(y_pred, np.ndarray) else y_pred

    abs_error = np.abs(y_pred - y_test)
    relative_error = abs_error / y_test

    print(f'{model_name} - MAE: {np.mean(abs_error):.4f}, MRE: {np.mean(relative_error):.4f}')

    return abs_error, relative_error
