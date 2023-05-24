#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Model effect module

File containing model effect function

"""

import os 
from typing import List, Union, Callable, Optional

import numpy as np
import pandas as pd
import polars as pl
import scienceplots
import matplotlib.pyplot as plt

import variables.prep_var as prep


# Convergence rate comparison function

def convergence_rate(losses: List[Union[np.ndarray, pl.DataFrame, pd.DataFrame]], models: Optional[List[str]] = ["Benchmark", "MLP"], colors: Optional[List[str]] = ["purple", "brown", "blue", "red"], folder: str = "photos", filename: Optional[str] = "convergence_rate.pdf", args: Optional[Callable] = None) -> None:
    
    """
    Plotted convergence rate (linear/log) for benchmark and MLP models

    Args:
        losses (List[Union[np.ndarray, pl.DataFrame, pd.DataFrame]]): Models losses
        models (Optional[List[str]], optional): Models names (default: ["Benchmark", "MLP"])
        colors (Optional[List[str]]): Colors names (default: ["purple", "brown", "blue", "red"])
        folder (str, optional): Sub-folder name in results folder (default: "photos")
        filename (str, optional): Parquet filename (default: "convergence_rate.pdf")
        args (Callable, optional): Arguments if you use run.py instead of tutorial.ipynb (default: None)

    Returns:
        None: Function does not return anything
    """
        
    # Default parameters
    default_params = {"dirpath": prep.DIRPATH}

    # Initialized parameters
    dict_args = {k: getattr(args, k, v) for k, v in default_params.items()}

    # Built plots
    plt.style.use(['science', 'ieee'])

    _, ax = plt.subplots(figsize=(18, 9))

    ax.grid(which='major', color='#999999', linestyle='--')
    ax.minorticks_on()
    ax.grid(which='minor', color='#999999', linestyle='--', alpha=0.25)

    for i, loss in enumerate(losses):

        loss = loss.to_numpy() if not isinstance(loss, np.ndarray) else loss

        ax.plot(loss[:, 0], label=f"{models[i]} Train Loss", color=colors[i])
        ax.plot(loss[:, 1], label=f"{models[i]} Validation Loss", color=colors[i+1])

        # Inset plot
        axins = ax.inset_axes([0.5, 0.30, 0.42, 0.48], transform=ax.transAxes)

        axins.semilogy(loss[:, 0], label=f"{models[i]} Train Loss", color=colors[i])
        axins.semilogy(loss[:, 1], label=f"{models[i]} Validation Loss", color=colors[i+1])

        axins.set_xlim([0, len(loss[:, 0])])
        axins.set_ylim([0.05, 0.5])
        axins.tick_params(axis='both', which='major', labelsize=12, pad=6)

        axins.legend(axins.get_legend_handles_labels()[0], axins.get_legend_handles_labels()[1], loc="best", fontsize=10)
        ax.indicate_inset_zoom(axins, alpha=0)

    ax.set_title('Convergence Rate', fontsize=16, pad=15)
    ax.set_xlabel("Epochs", fontsize=16, labelpad=15)
    ax.set_ylabel("Loss", fontsize=16, labelpad=15)
    ax.set_ylim([0, 1.2])

    ax.tick_params(axis='both', which='major', labelsize=14, pad=8)
    ax.legend(ax.get_legend_handles_labels()[0], ax.get_legend_handles_labels()[1], loc="best", fontsize=12)

    plt.rcParams.update({"text.usetex": True, "font.family": "serif", "pgf.texsystem": "pdflatex"})
    plt.savefig(os.path.join(dict_args['dirpath'], folder, filename), backend='pgf')
    plt.show()


# Errors boxplots function

def errors_boxplots(errors: List[np.ndarray] = None, label_names: List[str] = ["Benchmark", "MLP"], error_names: List[str] = ["$\eta$ Error", '$\eta$ Relative Error', '$\mu$ Error', '$\mu$ Relative Error'], folder: Optional[str] = "photos", filename: Optional[str] = "error_boxplots.pdf", args: Optional[Callable] = None) -> None:

    """
    Plotted absolute/relative errors boxplots for benchmark and MLP models

    Args:
        errors (List[np.ndarray], optional): Models eta and mu error / relative error (default: None)
        label_names (List[str], optional): Models names (default: ["Benchmark", "MLP"])
        error_names (List[str], optional): Errors names (default: ["$\eta$ Error", '$\eta$ Relative Error', '$\mu$ Error', '$\mu$ Relative Error'])
        folder (str, optional): Sub-folder name in results folder (default: "photos")
        filename (str, optional): Parquet filename (default: "error_boxplots.pdf")
        args (Callable, optional): Arguments if you use run.py instead of tutorial.ipynb (default: None)

    Returns:
        None: Function does not return anything
    """
        
    # Default parameters
    default_params = {"dirpath": prep.DIRPATH}

    # Initialized parameters
    dict_args = {k: getattr(args, k, v) for k, v in default_params.items()}

    # Converted to array
    errors = [error.to_numpy() if not isinstance(error, np.ndarray) else error for error in errors]

    # Regrouped errors and labels
    errors_list = list(map(np.ndarray.flatten, [error[:, i] for error in errors for i in range(error.shape[1])]))
    labels = [f"{label_name} {error_name}" for label_name in label_names for error_name in error_names]

    # Built boxplots
    plt.style.use(['science', 'ieee'])

    _, ax = plt.subplots(figsize=(30, 10))

    ax.grid(which='major', color='#999999', linestyle='--')
    ax.minorticks_on()
    ax.grid(which='minor', color='#999999', linestyle='--', alpha=0.25)

    ax.boxplot(errors_list, labels)

    ax.set_title('Error Comparison', fontsize=16, pad=15)
    ax.set_xlabel('Model', fontsize=16, labelpad=15)
    ax.set_ylabel('Error', fontsize=16, labelpad=15)
    ax.tick_params(axis='both', which='major', labelsize=14, pad=8)

    plt.rcParams.update({"text.usetex": True, "font.family": "serif", "pgf.texsystem": "pdflatex"})
    plt.savefig(os.path.join(dict_args['dirpath'], folder, filename), backend='pgf')
    plt.show()


# Variables effects boxplots function

def effects_boxplots(errors: List[np.ndarray] = None, errors_rel: List[np.ndarray] = None, label_names: List[str] = ["MLP"], 
                     error_names: List[str] = ["$\eta$ Error ($\Delta$ = 0.25)", "$\mu$ Error ($\Delta$ = 0.25)", "$\eta$ Error ($\Delta$ = 0.5)", "$\mu$ Error ($\Delta$ = 0.5)", "$\eta$ Error ($\Delta$ = 1.0)", "$\mu$ Error ($\Delta$ = 1.0)", "$\eta$ Error ($\Delta$ = 2.0)", "$\mu$ Error ($\Delta$ = 2.0)", "$\eta$ Error ($\Delta$ = 5.0)", "$\mu$ Error ($\Delta$ = 5.0)"], 
                     error_rel_names: List[str] = ["$\eta$ Relative Error ($\Delta$ = 0.25)", "$\mu$ Relative Error ($\Delta$ = 0.25)", "$\eta$ Relative Error ($\Delta$ = 0.5)", "$\mu$ Relative Error ($\Delta$ = 0.5)", "$\eta$ Relative Error ($\Delta$ = 1.0)", "$\mu$ Relative Error ($\Delta$ = 1.0)", "$\eta$ Relative Error ($\Delta$ = 2.0)", "$\mu$ Relative Error ($\Delta$ = 2.0)", "$\eta$ Relative Error ($\Delta$ = 5.0)", "$\mu$ Relative Error ($\Delta$ = 5.0)"],
                     folder: Optional[str] = "photos", filename: Optional[str] = "error_boxplots.pdf", args: Optional[Callable] = None) -> None:
    
    """
    Plotted absolute/relative errors boxplots to measure variables effects

    Args:
        errors (List[np.ndarray], optional): Models eta and mu errors (default: None)
        errors_rel (List[np.ndarray], optional): Models eta and mu relative errors (default: None)
        label_names (List[str], optional): Models names (default: ["Benchmark", "MLP"])
        error_names (List[str], optional): Errors names (default: ["$\eta$ Error ($\Delta$ = 0.25)", "$\mu$ Error ($\Delta$ = 0.25)", "$\eta$ Error ($\Delta$ = 0.5)", "$\mu$ Error ($\Delta$ = 0.5)", "$\eta$ Error ($\Delta$ = 1.0)", "$\mu$ Error ($\Delta$ = 1.0)", "$\eta$ Error ($\Delta$ = 2.0)", "$\mu$ Error ($\Delta$ = 2.0)", "$\eta$ Error ($\Delta$ = 5.0)", "$\mu$ Error ($\Delta$ = 5.0)"])
        error_rel_names (List[str], optional): Relative Errors names (default: ["$\eta$ Relative Error ($\Delta$ = 0.25)", "$\mu$ Relative Error ($\Delta$ = 0.25)", "$\eta$ Relative Error ($\Delta$ = 0.5)", "$\mu$ Relative Error ($\Delta$ = 0.5)", "$\eta$ Relative Error ($\Delta$ = 1.0)", "$\mu$ Relative Error ($\Delta$ = 1.0)", "$\eta$ Relative Error ($\Delta$ = 2.0)", "$\mu$ Relative Error ($\Delta$ = 2.0)", "$\eta$ Relative Error ($\Delta$ = 5.0)", "$\mu$ Relative Error ($\Delta$ = 5.0)"])
        folder (str, optional): Sub-folder name in results folder (default: "photos")
        filename (str, optional): Parquet filename (default: "error_boxplots.pdf")
        args (Callable, optional): Arguments if you use run.py instead of tutorial.ipynb (default: None)

    Returns:
        None: Function does not return anything
    """
        
    # Default parameters
    default_params = {"dirpath": prep.DIRPATH}

    # Initialized parameters
    dict_args = {k: getattr(args, k, v) for k, v in default_params.items()}

    # Converted to array
    errors = [error.to_numpy() if not isinstance(error, np.ndarray) else error for error in errors]
    errors_rel = [error_rel.to_numpy() if not isinstance(error_rel, np.ndarray) else error_rel for error_rel in errors_rel]

    # Regrouped errors and labels
    errors_list = list(map(np.ndarray.flatten, [error[:, i] for error in errors for i in range(error.shape[1])]))
    errors_rel_list = list(map(np.ndarray.flatten, [error_rel[:, i] for error_rel in errors_rel for i in range(error_rel.shape[1])]))
    labels_errors = [f"{label_name} {error_name}" for label_name in label_names for error_name in error_names]
    labels_errors_rel = [f"{label_name} {error_rel_name}" for label_name in label_names for error_rel_name in error_rel_names]

    # Built boxplots
    plt.style.use(['science', 'ieee'])
    
    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(35, 10))

    for ax in (ax1, ax2):
        ax.grid(which='major', color='#999999', linestyle='--')
        ax.minorticks_on()
        ax.grid(which='minor', color='#999999', linestyle='--', alpha=0.25)

    ax1.boxplot(errors_list, labels=labels_errors) 
    ax2.boxplot(errors_rel_list, labels=labels_errors_rel)

    ax1.set_title('Errors Comparison', fontsize=16, pad=15)
    ax2.set_title('Relative Errors Comparison', fontsize=16, pad=15)

    ax1.set_xlabel('Model', fontsize=16, labelpad=15)
    ax1.set_ylabel('Error', fontsize=16, labelpad=15)
    ax1.tick_params(axis='both', which='major', labelsize=14, pad=8)

    ax2.set_xlabel('Model', fontsize=16, labelpad=15)
    ax2.set_ylabel('Relative Error', fontsize=16, labelpad=15)
    ax2.tick_params(axis='both', which='major', labelsize=14, pad=8)

    plt.rcParams.update({"text.usetex": True, "font.family": "serif", "pgf.texsystem": "pdflatex"})
    plt.tight_layout(h_pad=3)
    plt.savefig(os.path.join(dict_args['dirpath'], folder, filename), backend='pgf')
    plt.show()