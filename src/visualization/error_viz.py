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
import seaborn as sns
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

def errors_boxplots(errors: List[np.ndarray] = None, label_names: List[str] = ["Benchmark", "MLP"], error_names: List[str] = ["$\eta$ Error", '$\eta$ Relative Error', '$\mu$ Error', '$\mu$ Relative Error'], showfliers: bool = True, folder: Optional[str] = "photos", filename: Optional[str] = "error_boxplots.pdf", args: Optional[Callable] = None) -> None:

    """
    Plotted absolute/relative errors boxplots for benchmark and MLP models

    Args:
        errors (List[np.ndarray], optional): Models eta and mu error / relative error (default: None)
        label_names (List[str], optional): Models names (default: ["Benchmark", "MLP"])
        error_names (List[str], optional): Errors names (default: ["$\eta$ Error", '$\eta$ Relative Error', '$\mu$ Error', '$\mu$ Relative Error'])
        showfliers (bool, optional): Show outliers (default: True)
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

    ax.boxplot(errors_list, labels=labels, showfliers=showfliers)

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
                     showfliers: bool = True, folder: Optional[str] = "photos", filename: Optional[str] = "error_boxplots.pdf", args: Optional[Callable] = None) -> None:
    
    """
    Plotted absolute/relative errors boxplots to measure variables effects

    Args:
        errors (List[np.ndarray], optional): Models eta and mu errors (default: None)
        errors_rel (List[np.ndarray], optional): Models eta and mu relative errors (default: None)
        label_names (List[str], optional): Models names (default: ["Benchmark", "MLP"])
        error_names (List[str], optional): Errors names (default: ["$\eta$ Error ($\Delta$ = 0.25)", "$\mu$ Error ($\Delta$ = 0.25)", "$\eta$ Error ($\Delta$ = 0.5)", "$\mu$ Error ($\Delta$ = 0.5)", "$\eta$ Error ($\Delta$ = 1.0)", "$\mu$ Error ($\Delta$ = 1.0)", "$\eta$ Error ($\Delta$ = 2.0)", "$\mu$ Error ($\Delta$ = 2.0)", "$\eta$ Error ($\Delta$ = 5.0)", "$\mu$ Error ($\Delta$ = 5.0)"])
        error_rel_names (List[str], optional): Relative Errors names (default: ["$\eta$ Relative Error ($\Delta$ = 0.25)", "$\mu$ Relative Error ($\Delta$ = 0.25)", "$\eta$ Relative Error ($\Delta$ = 0.5)", "$\mu$ Relative Error ($\Delta$ = 0.5)", "$\eta$ Relative Error ($\Delta$ = 1.0)", "$\mu$ Relative Error ($\Delta$ = 1.0)", "$\eta$ Relative Error ($\Delta$ = 2.0)", "$\mu$ Relative Error ($\Delta$ = 2.0)", "$\eta$ Relative Error ($\Delta$ = 5.0)", "$\mu$ Relative Error ($\Delta$ = 5.0)"])
        showfliers (bool, optional): Show outliers (default: True)
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

    ax1.boxplot(errors_list, labels=labels_errors, showfliers=showfliers) 
    ax2.boxplot(errors_rel_list, labels=labels_errors_rel, showfliers=showfliers)

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


# Intensity reconstruction plot function

def reconstruction_plot(decoded_intensities: List[np.ndarray], integrated_intensities: List[np.ndarray], label_names: List[str] = ["Decoded Intensity", "Integrated Intensity"], 
                        params_names: List[str] = [[1.0, 0.2], [3.0, 0.2], [1.0, 0.7], [3.0, 0.7]], folder: Optional[str] = "photos", filename: Optional[str] = "reconstruction_plot.pdf", args: Optional[Callable] = None) -> None:

    """
    Plot intensities reconstruction and NRMSE

    Args:
        decoded_intensities (List[np.ndarray]): Decoded intensity list
        integrated_intensities (List[np.ndarray]): Integrated intensity list
        label_names (List[str], optional): Intensity type (default: ["Decoded Intensity", "Integrated Intensity"])
        params_names (List[str], optional): Testing parameters (default: [[1.0, 0.2], [3.0, 0.2], [1.0, 0.7], [3.0, 0.7]])
        folder (str, optional): Sub-folder name in results folder (default: "photos")
        filename (str, optional): Parquet filename (default: "reconstruction_plot.pdf")
        args (Callable, optional): Arguments if you use run.py instead of tutorial.ipynb (default: None)

    Returns:
        None: Function does not return anything
    """
        
    # Default parameters
    default_params = {"dirpath": prep.DIRPATH}

    # Initialized parameters
    dict_args = {k: getattr(args, k, v) for k, v in default_params.items()}

    # Converted to array
    decoded_intensities = [decoded_intensity.to_numpy() if not isinstance(decoded_intensity, np.ndarray) else decoded_intensity for decoded_intensity in decoded_intensities]
    integrated_intensities = [integrated_intensity.to_numpy() if not isinstance(integrated_intensity, np.ndarray) else integrated_intensity for integrated_intensity in integrated_intensities]

    # Built lineplots + NRMSE
    plt.style.use(['science', 'ieee'])
    
    # factors = [3.075, 2.53, 3.35, 1.8]

    _, axes = plt.subplots(len(decoded_intensities), 1, figsize=(42, 24))
    
    for ax, decoded, integrated, params_name, factor in zip(axes, decoded_intensities, integrated_intensities, params_names, factors):
        
        ax.grid(which='major', color='#999999', linestyle='--')
        ax.minorticks_on()
        ax.grid(which='minor', color='#999999', linestyle='--', alpha=0.25)

        # decoded = decoded * factor

        ax.plot(range(len(decoded)), decoded, color='red', label=label_names[0])
        ax.plot(range(len(integrated)), integrated, color='blue', label=label_names[1])

        nrmse = np.sqrt(np.mean((decoded - integrated)**2)) / (np.max(integrated) - np.min(integrated))
        
        ax.set_title(r"Intensity Reconstruction ($\beta$ = {0}, $\eta$ = {1}, NRMSE = {2:.4f})".format(params_name[0], params_name[1], nrmse), fontsize=16, pad=15)
        ax.set_xlabel("Time", fontsize=16, labelpad=15)
        ax.set_ylabel("Intensity", fontsize=16, labelpad=15)
        ax.tick_params(axis="both", which="major", labelsize=14, pad=8)
        ax.legend(loc="best", fontsize=12)

    plt.rcParams.update({"text.usetex": True, "font.family": "serif", "pgf.texsystem": "pdflatex"})
    plt.tight_layout(pad=4, h_pad=4)
    plt.savefig(os.path.join(dict_args['dirpath'], folder, filename), backend='pgf')
    plt.show()


# Density plot function

def density_plot(encoded_parameters: List[np.ndarray], decoded_parameters: List[np.ndarray], params_names: List[str] = [[1.0, 0.2], [3.0, 0.2], [1.0, 0.7], [3.0, 0.7]],
                 folder: Optional[str] = "photos", filename: Optional[str] = "density_plot.pdf", args: Optional[Callable] = None) -> None:

    """
     Plot density of hawkes process parameters

    Args:
        encoded_parameters (List[np.ndarray]): Encoded parameters list
        decoded_parameters (List[np.ndarray]): Decoded parameters list
        params_names (List[str], optional): Testing parameters (default: [[1.0, 0.2], [3.0, 0.2], [1.0, 0.7], [3.0, 0.7]])
        folder (str, optional): Sub-folder name in results folder (default: "photos")
        filename (str, optional): Parquet filename (default: "density_plot.pdf")
        args (Callable, optional): Arguments if you use run.py instead of tutorial.ipynb (default: None)

    Returns:
        None: Function does not return anything
    """
        
    # Default parameters
    default_params = {"dirpath": prep.DIRPATH}

    # Initialized parameters
    dict_args = {k: getattr(args, k, v) for k, v in default_params.items()}

    # Converted to array
    encoded_parameters = [encoded_parameters.to_numpy() if not isinstance(encoded_parameter, np.ndarray) else encoded_parameter for encoded_parameter in encoded_parameters]
    decoded_parameters = [decoded_parameter.to_numpy() if not isinstance(decoded_parameter, np.ndarray) else decoded_parameter for decoded_parameter in decoded_parameters]

    # Built desnity plots + NRMSE
    plt.style.use(['science', 'ieee'])

    _, axes = plt.subplots(len(decoded_parameters), 1, figsize=(42, 24))
    
    for ax, encoded_parameter, decoded_parameter, params_name in zip(axes, encoded_parameters, decoded_parameters, params_names):
        
        ax.grid(which='major', color='#999999', linestyle='--')
        ax.minorticks_on()
        ax.grid(which='minor', color='#999999', linestyle='--', alpha=0.25)

        sns.kdeplot(data=decoded_parameter, x="eta_pred", y="mu_pred", fill=True, cmap="viridis", levels=5)

        ax.axhline(y=encoded_parameter[:, 1], color="sienna2", linewidth=1)
        ax.axvline(x=encoded_parameter[:, 0], color="sienna2", linewidth=1)
        ax.scatter(x=encoded_parameter[:, 0], y=encoded_parameter[:, 1], color="sienna2", s=25)
        ax.text(0.15, 3.8, "True value", color="sienna2")

        ax.set_title(r"Density Estimation ($\beta$ = {0}, $\eta$ = {1})".format(params_name[0], params_name[1]), fontsize=16, pad=15)
        ax.set_xlabel(r"$\eta$", fontsize=16, labelpad=15)
        ax.set_ylabel(r"$\beta$", fontsize=16, labelpad=15)
        ax.tick_params(axis="both", which="major", labelsize=14, pad=8)
        ax.legend(loc="best", fontsize=12)

    plt.rcParams.update({"text.usetex": True, "font.family": "serif", "pgf.texsystem": "pdflatex"})
    plt.tight_layout(pad=4, h_pad=4)
    plt.savefig(os.path.join(dict_args['dirpath'], folder, filename), backend='pgf')
    plt.show()

