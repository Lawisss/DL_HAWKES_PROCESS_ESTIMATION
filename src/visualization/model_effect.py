#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Model effect module

File containing model effect function

"""

import os 
from typing import Callable, Optional

import numpy as np
import scienceplots
import matplotlib.pyplot as plt

import variables.prep_var as prep


# Error boxplots function

def error_boxplots(bench_abs_error: np.ndarray, bench_rel_error: np.ndarray, mlp_abs_error: np.ndarray, mlp_rel_error: np.ndarray, folder: str = "photos", filename: str = "error_boxplots.pdf", args: Optional[Callable] = None) -> None:

    """
    Plotted absolute/relative error boxplots for benchmark and MLP models

    Args:
        bench_abs_error (np.ndarray): Benchmark model absolute error
        bench_rel_error (np.ndarray): Benchmark model relative error
        mlp_abs_error (np.ndarray): MLP model absolute error
        mlp_rel_error (np.ndarray): MLP model relative error
        folder (str, optional): Sub-folder name in results folder (default: "photos")
        filename (str, optional): Parquet filename (default: "photos")
        args (Callable, optional): Arguments if you use run.py instead of tutorial.ipynb (default: None)

    Returns:
        Absolute error and relative error
    """
        
    # Default parameters
    default_params = {"dirpath": prep.DIRPATH}

    # Initialized parameters
    dict_args = {k: getattr(args, k, v) for k, v in default_params.items()}

    # Regrouped errors
    abs_errors = list(map(np.ndarray.flatten, [bench_abs_error, mlp_abs_error]))
    rel_errors = list(map(np.ndarray.flatten, [bench_rel_error, mlp_rel_error]))

    # Built boxplots
    plt.style.use(['science', 'ieee'])

    _, ax = plt.subplots(figsize=(16, 8))

    ax.grid(which='major', color='#999999', linestyle='--')
    ax.minorticks_on()
    ax.grid(which='minor', color='#999999', linestyle='--', alpha=0.25)

    ax.boxplot(abs_errors + rel_errors, labels=['Benchmark Absolute Error', 'MLP Absolute Error', 
                                                'Benchmark Relative Error', 'MLP Relative Error'])

    ax.set_title('Error Comparison', fontsize=16, pad=15)
    ax.set_xlabel('Model', fontsize=16, labelpad=15)
    ax.set_ylabel('Error', fontsize=16, labelpad=15)
    ax.tick_params(axis='both', which='major', labelsize=14, pad=8)

    plt.savefig(os.path.join(dict_args['dirpath'], folder, filename))
    plt.show()