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

    ax.set_title('Error Comparison', fontsize=33, pad=22)
    ax.set_xlabel('Model', fontsize=33, labelpad=22)
    ax.set_ylabel('Error', fontsize=33, labelpad=22)
    ax.tick_params(axis='both', which='major', labelsize=26, pad=10)

    plt.rcParams.update({"text.usetex": True, "font.family": "serif", "pgf.texsystem": "pdflatex"})
    plt.savefig(os.path.join(dict_args['dirpath'], folder, filename), backend='pgf')
    plt.show()


# Variables effects boxplots function

def effects_boxplots(eta_errors = None, eta_errors_rel = None, mu_errors = None, mu_errors_rel = None, label_names = ["MLE"], 
                     eta_error_names = ["$\eta$ Error ($\Delta$ = 0.25)", "$\eta$ Error ($\Delta$ = 0.5)", "$\eta$ Error ($\Delta$ = 1.0)", "$\eta$ Error ($\Delta$ = 2.0)", "$\eta$ Error ($\Delta$ = 5.0)"], 
                     eta_error_rel_names = ["$\eta$ Relative Error ($\Delta$ = 0.25)", "$\eta$ Relative Error ($\Delta$ = 0.5)", "$\eta$ Relative Error ($\Delta$ = 1.0)", "$\eta$ Relative Error ($\Delta$ = 2.0)", "$\eta$ Relative Error ($\Delta$ = 5.0)"],
                     mu_error_names = ["$\mu$ Error ($\Delta$ = 0.25)", "$\mu$ Error ($\Delta$ = 0.5)", "$\mu$ Error ($\Delta$ = 1.0)", "$\mu$ Error ($\Delta$ = 2.0)", "$\mu$ Error ($\Delta$ = 5.0)"],
                     mu_error_rel_names = ["$\mu$ Relative Error ($\Delta$ = 0.25)", "$\mu$ Relative Error ($\Delta$ = 0.5)", "$\mu$ Relative Error ($\Delta$ = 1.0)", "$\mu$ Relative Error ($\Delta$ = 2.0)", "$\mu$ Relative Error ($\Delta$ = 5.0)"],
                     showfliers: bool = True, folder = "photos", filename = "error_boxplots.pdf", args = None) -> None:
    
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
    eta_errors = [eta_error.to_numpy() if not isinstance(eta_error, np.ndarray) else eta_error for eta_error in eta_errors]
    eta_errors_rel = [eta_error_rel.to_numpy() if not isinstance(eta_error_rel, np.ndarray) else eta_error_rel for eta_error_rel in eta_errors_rel]
    mu_errors = [mu_error.to_numpy() if not isinstance(mu_error, np.ndarray) else mu_error for mu_error in mu_errors]
    mu_errors_rel = [mu_error_rel.to_numpy() if not isinstance(mu_error_rel, np.ndarray) else mu_error_rel for mu_error_rel in mu_errors_rel]

    # Regrouped errors and labels
    eta_errors_list = list(map(np.ndarray.flatten, [eta_error[:, i] for eta_error in eta_errors for i in range(eta_error.shape[1])]))
    eta_errors_rel_list = list(map(np.ndarray.flatten, [eta_error_rel[:, i] for eta_error_rel in eta_errors_rel for i in range(eta_error_rel.shape[1])]))
    mu_errors_list = list(map(np.ndarray.flatten, [mu_error[:, i] for mu_error in mu_errors for i in range(mu_error.shape[1])]))
    mu_errors_rel_list = list(map(np.ndarray.flatten, [mu_error_rel[:, i] for mu_error_rel in mu_errors_rel for i in range(mu_error_rel.shape[1])]))
    eta_labels_errors = [f"{label_name} {eta_error_name}" for label_name in label_names for eta_error_name in eta_error_names]
    eta_labels_errors_rel = [f"{label_name} {eta_error_rel_name}" for label_name in label_names for eta_error_rel_name in eta_error_rel_names]
    mu_labels_errors = [f"{label_name} {mu_error_name}" for label_name in label_names for mu_error_name in mu_error_names]
    mu_labels_errors_rel = [f"{label_name} {mu_error_rel_name}" for label_name in label_names for mu_error_rel_name in mu_error_rel_names]

    # Built boxplots
    plt.style.use(['science', 'ieee'])
    
    _, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(45, 35))

    for ax in (ax1, ax2, ax3, ax4):
        ax.grid(which='major', color='#999999', linestyle='--')
        ax.minorticks_on()
        ax.grid(which='minor', color='#999999', linestyle='--', alpha=0.25)

    ax1.boxplot(eta_errors_list, labels=eta_labels_errors, showfliers=showfliers) 
    ax2.boxplot(eta_errors_rel_list, labels=eta_labels_errors_rel, showfliers=showfliers)
    ax3.boxplot(mu_errors_list, labels=mu_labels_errors, showfliers=showfliers) 
    ax4.boxplot(mu_errors_rel_list, labels=mu_labels_errors_rel, showfliers=showfliers)

    ax1.set_title('Errors Comparison ($\eta$)', fontsize=40, pad=28)
    ax2.set_title('Relative Errors Comparison ($\eta$)', fontsize=40, pad=28)
    ax3.set_title('Errors Comparison ($\mu$)', fontsize=40, pad=28)
    ax4.set_title('Relative Errors Comparison ($\mu$)', fontsize=40, pad=28)

    ax1.set_xlabel('Model', fontsize=40, labelpad=28)
    ax1.set_ylabel('Error', fontsize=40, labelpad=28)
    ax1.tick_params(axis='both', which='major', labelsize=33, pad=12)

    ax2.set_xlabel('Model', fontsize=40, labelpad=28)
    ax2.set_ylabel('Relative Error', fontsize=40, labelpad=28)
    ax2.tick_params(axis='both', which='major', labelsize=33, pad=12)

    ax3.set_xlabel('Model', fontsize=40, labelpad=28)
    ax3.set_ylabel('Relative Error', fontsize=40, labelpad=28)
    ax3.tick_params(axis='both', which='major', labelsize=26, pad=12)

    ax4.set_xlabel('Model', fontsize=40, labelpad=28)
    ax4.set_ylabel('Relative Error', fontsize=40, labelpad=28)
    ax4.tick_params(axis='both', which='major', labelsize=26, pad=12)

    plt.rcParams.update({"text.usetex": True, "font.family": "serif", "pgf.texsystem": "pdflatex"})
    plt.tight_layout(h_pad=3)
    plt.savefig(os.path.join(dict_args['dirpath'], folder, filename), backend='pgf')
    plt.show()


# Predictions boxplots function

def pred_boxplots(mle_preds: List[np.ndarray], mlp_preds: List[np.ndarray], lstm_preds: List[np.ndarray], eta_true: Optional[float] = 0.2, median_true: Optional[float] = None, deltas: List[float] = [0.25, 0.5, 1.0, 2.0, 5.0], labels = ['MLE', 'MLP', 'LSTM'], 
                  showfliers: bool = True, folder: Optional[str] = "photos", filename: Optional[str] = "pred_boxplots.pdf", args: Optional[Callable] = None) -> None:
    
    """
    Plotted eta/mu predictions boxplots to compare models

    Args:
        mle_preds (List[np.ndarray], optional): MLE predictions
        mlp_preds (List[np.ndarray], optional): MLP predictions
        lstm_preds (List[np.ndarray]): LSTM predictions
        eta_true (Optional[float]): Branching ratio true value (default: 0.2)
        median_true (Optional[float]): Median true value (default: None)
        deltas (List[float], optional): Discretisation step values (default: [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10])
        labels (List[str], optional): Models names (default: ["MLE", "MLP", "LSTM"])
        showfliers (bool, optional): Show outliers (default: True)
        folder (str, optional): Sub-folder name in results folder (default: "photos")
        filename (str, optional): Parquet filename (default: "pred_boxplots.pdf")
        args (Callable, optional): Arguments if you use run.py instead of tutorial.ipynb (default: None)

    Returns:
        None: Function does not return anything
    """
        
    # Default parameters
    default_params = {"dirpath": prep.DIRPATH}

    # Initialized parameters
    dict_args = {k: getattr(args, k, v) for k, v in default_params.items()}

    # Converted to array
    mle_preds = [mle_pred.to_numpy().flatten() if not isinstance(mle_pred, np.ndarray) else mle_pred for mle_pred in mle_preds]
    mlp_preds = [mlp_pred.to_numpy().flatten() if not isinstance(mlp_pred, np.ndarray) else mlp_pred for mlp_pred in mlp_preds]
    lstm_preds = [lstm_pred.to_numpy().flatten() if not isinstance(lstm_pred, np.ndarray) else lstm_pred for lstm_pred in lstm_preds]

    # Built boxplots
    plt.style.use(['science', 'ieee'])

    _, ax = plt.subplots(figsize=(40, 18))
    sns.set_palette("muted")

    ax.grid(which='major', color='#999999', linestyle='--')
    ax.minorticks_on()
    ax.grid(which='minor', color='#999999', linestyle='--', alpha=0.25)

    positions = np.arange(len(mle_preds)) * (len(labels) + 1)
    colors = sns.color_palette()

    boxplots = [ax.boxplot(preds, positions=positions + i * 0.5 - 0.6, widths=0.4, showfliers=showfliers, patch_artist=True, medianprops={"color": "red"}, whis=[20, 80]) for i, preds in enumerate([mle_preds, mlp_preds, lstm_preds])]
    [plt.setp(boxplot[component], color=color) for boxplot, color in zip(boxplots, colors) for component in ["boxes"]]
    legend_labels = [ax.plot([], marker='s', markersize=10, markerfacecolor=color, label=label)[0] for label, color in zip(labels, colors)]

    ax.set_xticks(positions)
    ax.set_yticks(ax.get_yticks())
    ax.set_xticklabels(deltas, fontsize=33)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=33)
    
    # Branching ratio ylim: bottom=0.06, top=0.36, bottom=0.23, top=0.6, bottom=0.56, top=0.87
    # Baseline intensity ylim: bottom=3.5, top=4.8, bottom=0.5, top=3.9, bottom=0.2, top=1.5

    # ax.set_ylim(bottom=3.5, top=4.8) 
    ax.tick_params(axis='both', which='both', pad=10)
    ax.axhline(y=median_true, color='orange', linestyle='--')
    ax.legend(handles=legend_labels, labels=labels, loc="best", fontsize=28)
    
    # Branching ratio / Baseline Intensity
    ax.set_title(r'Predictions boxplots ($\eta$ = {0}, $\beta$ = 2.0)'.format(eta_true), fontsize=33, pad=22)
    ax.set_xlabel(r'Discretisation step ($\Delta$)', fontsize=33, labelpad=22)
    ax.set_ylabel(r'Branching ratio predictions ($\hat{\eta})$', fontsize=33, labelpad=22)

    plt.rcParams.update({"text.usetex": True, "font.family": "serif", "pgf.texsystem": "pdflatex"})
    plt.tight_layout()
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
    
    for ax, decoded, integrated, params_name, factor in zip(axes, decoded_intensities, integrated_intensities, params_names):
        
        ax.grid(which='major', color='#999999', linestyle='--')
        ax.minorticks_on()
        ax.grid(which='minor', color='#999999', linestyle='--', alpha=0.25)

        # decoded = decoded * factor

        ax.plot(range(len(decoded)), decoded, color='red', label=label_names[0])
        ax.plot(range(len(integrated)), integrated, color='blue', label=label_names[1])

        nrmse = np.sqrt(np.mean((decoded - integrated)**2)) / (np.max(integrated) - np.min(integrated))
        
        ax.set_title(r"Intensity Reconstruction ($\beta$ = {0}, $\eta$ = {1}, NRMSE = {2:.4f})".format(params_name[0], params_name[1], nrmse), fontsize=33, pad=22)
        ax.set_xlabel("Time", fontsize=33, labelpad=22)
        ax.set_ylabel("Intensity", fontsize=33, labelpad=22)
        ax.tick_params(axis="both", which="major", labelsize=26, pad=10)
        ax.legend(loc="best", fontsize=28)

    plt.rcParams.update({"text.usetex": True, "font.family": "serif", "pgf.texsystem": "pdflatex"})
    plt.tight_layout(pad=5, h_pad=5)
    plt.savefig(os.path.join(dict_args['dirpath'], folder, filename), backend='pgf')
    plt.show()


# NRMSE Boxplot function

def nrmse_boxplot(vae_errors: List[np.ndarray], dd_errors: List[np.ndarray], labels: List[str] = ['Poisson-VAE', 'Dueling Decoder'], xlabels: List[str] = [r"($\beta$ = 1.0, $\eta$ = 0.2)", r"($\beta$ = 3.0, $\eta$ = 0.2)", r"($\beta$ = 1.0, $\eta$ = 0.7)", r"($\beta$ = 3.0, $\eta$ = 0.7)"], 
                  showfliers: bool = True, folder: Optional[str] = "photos", filename: Optional[str] = "nrmse_boxplot.pdf", args: Optional[Callable] = None) -> None:
    
    """
    Computed NRMSE error

    Args:
        vae_errors (List[np.ndarray]): Poisson-VAE NRMSE errors
        dd_errors (List[np.ndarray]): Dueling Decoder NRMSE errors
        labels (List[str]): Models labels (default: ['Poisson-VAE', 'Dueling Decoder'])
        xlabels (List[str]): Parameters labels  (default: ["($\beta$ = 1.0, $\eta$ = 0.2)", "($\beta$ = 3.0, $\eta$ = 0.2)",  "($\beta$ = 1.0, $\eta$ = 0.7)", "($\beta$ = 3.0, $\eta$ = 0.7)"])
        showfliers (bool, optional): Show outliers (default: True)
        folder (str, optional): Sub-folder name in results folder (default: "photos")
        filename (str, optional): Parquet filename to save results (default: "nrmse_boxplot.parquet")
        args (Callable, optional): Arguments if you use run.py instead of tutorial.ipynb (default: None)

    Returns:
        None: Function does not return anything
    """

    # Default parameters
    default_params = {"dirpath": prep.DIRPATH}

    # Initialized parameters
    dict_args = {k: getattr(args, k, v) for k, v in default_params.items()}

    # Converted to array
    vae_errors = [vae_error.to_numpy().flatten() if not isinstance(vae_error, np.ndarray) else vae_error for vae_error in vae_errors]
    dd_errors = [dd_error.to_numpy().flatten() if not isinstance(dd_error, np.ndarray) else dd_error for dd_error in dd_errors]

    # Built lineplots + NRMSE
    plt.style.use(['science', 'ieee'])

    _, ax = plt.subplots(figsize=(18, 9))
    sns.set_palette("muted")

    ax.grid(which='major', color='#999999', linestyle='--')
    ax.minorticks_on()
    ax.grid(which='minor', color='#999999', linestyle='--', alpha=0.25)
    
    positions = np.arange(len(vae_errors)) * (len(xlabels) + 1)
    colors = sns.color_palette()

    boxplots = [ax.boxplot(preds, positions=positions + i * 0.5 - 0.6, widths=0.4, showfliers=showfliers, patch_artist=True, medianprops={"color": "red"}, whis=[20, 80]) for i, preds in enumerate([vae_errors, dd_errors])]
    [plt.setp(boxplot[component], color=color) for boxplot, color in zip(boxplots, colors) for component in ["boxes"]]
    legend_labels = [ax.plot([], marker='s', markersize=10, markerfacecolor=color, label=label)[0] for label, color in zip(labels, colors)]
    
    ax.set_xticks(positions * 0.96)
    ax.set_yticks(ax.get_yticks())
    ax.set_xticklabels(xlabels, fontsize=33)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=33)

    ax.set_title('Reconstruction Error Comparison', fontsize=33, pad=22)
    ax.set_xlabel('Parameters', fontsize=33, labelpad=22)
    ax.set_ylabel('Error (NRMSE)', fontsize=33, labelpad=22)
    ax.tick_params(axis='both', which='major', labelsize=26, pad=10)
    ax.legend(handles=legend_labels, labels=labels, loc="best", fontsize=28)

    plt.rcParams.update({"text.usetex": True, "font.family": "serif", "pgf.texsystem": "pdflatex"})
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

        ax.set_title(r"Density Estimation ($\beta$ = {0}, $\eta$ = {1})".format(params_name[0], params_name[1]), fontsize=33, pad=22)
        ax.set_xlabel(r"$\eta$", fontsize=33, labelpad=22)
        ax.set_ylabel(r"$\beta$", fontsize=33, labelpad=22)
        ax.tick_params(axis="both", which="major", labelsize=26, pad=10)
        ax.legend(loc="best", fontsize=28)

    plt.rcParams.update({"text.usetex": True, "font.family": "serif", "pgf.texsystem": "pdflatex"})
    plt.tight_layout(pad=5, h_pad=5)
    plt.savefig(os.path.join(dict_args['dirpath'], folder, filename), backend='pgf')
    plt.show()

