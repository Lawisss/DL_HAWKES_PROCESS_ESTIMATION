#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Evaluation module

File containing model evaluation functions

"""

from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import polars as pl

from tools.utils import write_parquet


# Error / Relative error function

def compute_errors(y_test: Union[np.ndarray, pl.DataFrame, pd.DataFrame], y_pred: Union[np.ndarray, pl.DataFrame, pd.DataFrame], model_name: Optional[str] = "Benchmark", record: bool = True, filename: Optional[str] = "errors.parquet") -> Tuple[np.ndarray, np.ndarray]:
    
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
    if record:
        write_parquet(pl.DataFrame(np.column_stack((eta_error, eta_rel_error, mu_error, mu_rel_error)), schema=["eta_error", "eta_rel_error", "mu_error", "mu_rel_error"]), filename=filename)

    return pl.DataFrame(np.column_stack((eta_error, eta_rel_error, mu_error, mu_rel_error)), schema=["eta_error", "eta_rel_error", "mu_error", "mu_rel_error"])
