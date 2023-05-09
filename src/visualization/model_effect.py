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

# def convergence_rate(bench_losses: Optional[Union[np.ndarray, pl.DataFrame, pd.DataFrame]], mlp_losses: Optional[Union[np.ndarray, pl.DataFrame, pd.DataFrame]], folder: Optional[str] = "photos", filename: Optional[str] = "convergence_rate.pdf", args: Optional[Callable] = None) -> None:

#     """
#     Plotted convergence rate (linear/log) for benchmark and MLP models

#     Args:
#         mlp_losses (Union[np.ndarray, pl.DataFrame, pd.DataFrame]): MLP model losses
#         bench_losses (Union[np.ndarray, pl.DataFrame, pd.DataFrame], optional): Benchmark model losses (default: None)
#         folder (str, optional): Sub-folder name in results folder (default: "photos")
#         filename (str, optional): Parquet filename (default: "convergence_rate.pdf")
#         args (Callable, optional): Arguments if you use run.py instead of tutorial.ipynb (default: None)

#     Returns:
#         None: Function does not return anything
#     """

#     # Checked types
#     bench_losses = bench_losses.to_numpy() if not isinstance(bench_losses, np.ndarray) else bench_losses
#     mlp_losses = mlp_losses.to_numpy() if not isinstance(mlp_losses, np.ndarray) else mlp_losses

#     # Default parameters
#     default_params = {"dirpath": prep.DIRPATH}

#     # Initialized parameters
#     dict_args = {k: getattr(args, k, v) for k, v in default_params.items()}

#     # Built plots
#     plt.style.use(['science', 'ieee'])

#     _, ax = plt.subplots(figsize=(18, 9))

#     ax.grid(which='major', color='#999999', linestyle='--')
#     ax.minorticks_on()
#     ax.grid(which='minor', color='#999999', linestyle='--', alpha=0.25)

#     ax.plot(bench_losses[:, 0], label="Benchmark Train Loss", color='purple')
#     ax.plot(bench_losses[:, 1], label="Benchmark Validation Loss", color='brown')
#     ax.plot(mlp_losses[:, 0], label="MLP Train Loss", color='blue')
#     ax.plot(mlp_losses[:, 1], label="MLP Validation Loss", color='red')

#     ax.set_title('Convergence Rate', fontsize=16, pad=15)
#     ax.set_xlabel("Epochs", fontsize=16, labelpad=15)
#     ax.set_ylabel("Loss", fontsize=16, labelpad=15)
#     ax.set_ylim([0, 1.2])

#     ax.tick_params(axis='both', which='major', labelsize=14, pad=8)
#     ax.legend(ax.get_legend_handles_labels()[0], ax.get_legend_handles_labels()[1], loc="best", fontsize=12)

#     # Inset plot
#     axins = ax.inset_axes([0.5, 0.30, 0.42, 0.48], transform=ax.transAxes)

#     axins.semilogy(bench_losses[:, 0], label="Benchmark Train Loss", color='cyan')
#     axins.semilogy(bench_losses[:, 1], label="Benchmark Validation Loss", color='magenta')
#     axins.semilogy(mlp_losses[:, 0], label="MLP Train Loss", color='green')
#     axins.semilogy(mlp_losses[:, 1], label="MLP Validation Loss", color='orange')

#     axins.set_xlim([0, len(bench_losses[:, 0])])
#     axins.set_ylim([0.08, 0.8])

#     axins.tick_params(axis='both', which='major', labelsize=12, pad=6)
#     axins.legend(axins.get_legend_handles_labels()[0], axins.get_legend_handles_labels()[1], loc="best", fontsize=10)
#     ax.indicate_inset_zoom(axins, alpha=0)

#     plt.rcParams.update({"text.usetex": True, "font.family": "serif", "pgf.texsystem": "pdflatex"})
#     plt.savefig(os.path.join(dict_args['dirpath'], folder, filename), backend='pgf')
#     plt.show()


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

        ax.plot(loss[:, 0], label=models[i] + ' Train Loss', color=colors[i])
        ax.plot(loss[:, 1], label=models[i] + ' Validation Loss', color=colors[i+1])

        # Inset plot
        axins = ax.inset_axes([0.5, 0.30, 0.42, 0.48], transform=ax.transAxes)

        axins.semilogy(loss[:, 0], label=models[i] + ' Train Loss', color=colors[i])
        axins.semilogy(loss[:, 1], label=models[i] + ' Validation Loss', color=colors[i+1])

        axins.set_xlim([0, len(loss[:, 0])])
        axins.set_ylim([0.08, 0.8])
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


# Error boxplots function

def error_boxplots(errors: List[np.ndarray] = None, folder: Optional[str] = "photos", filename: Optional[str] = "error_boxplots.pdf", args: Optional[Callable] = None) -> None:

    """
    Plotted absolute/relative error boxplots for benchmark and MLP models

    Args:
        errors (List[np.ndarray]): Models eta and mu error / relative error
        mlp_eta_errors (np.ndarray): MLP model eta error / relative error
        mlp_mu_errors (np.ndarray): MLP model mu error / relative error
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

    # Regrouped errors
    error_groups = list(map(lambda x: list(map(np.ndarray.flatten, [x[:, 0], x[:, 1], x[:, 2], x[:, 3]])), errors))

    # Built boxplots
    plt.style.use(['science', 'ieee'])

    _, ax = plt.subplots(figsize=(30, 10))

    ax.grid(which='major', color='#999999', linestyle='--')
    ax.minorticks_on()
    ax.grid(which='minor', color='#999999', linestyle='--', alpha=0.25)

    ax.boxplot(eta_error + eta_rel_error + mu_error + mu_rel_error, labels=['Benchmark $\eta$ Error', 'MLP $\eta$ Error', 
                                                                            'Benchmark $\eta$ Relative Error', 'MLP $\eta$ Relative Error',
                                                                            'Benchmark $\mu$ Error', 'MLP $\mu$ Error',
                                                                            'Benchmark $\mu$ Relative Error', 'MLP $\mu$ Relative Error'])

    ax.set_title('Error Comparison', fontsize=16, pad=15)
    ax.set_xlabel('Model', fontsize=16, labelpad=15)
    ax.set_ylabel('Error', fontsize=16, labelpad=15)
    ax.tick_params(axis='both', which='major', labelsize=14, pad=8)

    plt.rcParams.update({"text.usetex": True, "font.family": "serif", "pgf.texsystem": "pdflatex"})
    plt.savefig(os.path.join(dict_args['dirpath'], folder, filename), backend='pgf')
    plt.show()