#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Discretization step effect module

File containing discretization step effect function

"""

import os
from typing import List, Callable, Optional

import numpy as np
import polars as pl
import matplotlib.pyplot as plt

import variables.hawkes_var as hwk
from dl.mlp_model import MLPTrainer
from tools.utils import write_parquet
from hawkes.simulation import hawkes_simulations
from hawkes.hyperparameters import hyper_params_simulation


def delta_effect(val_x, deltas: List = [0.25, 0.5, 1.0, 2.0, 5.0], number_of_tests: int = 100, folder: str = "photos", filename: str = "activity_effect.pdf", args: Optional[Callable] = None):

    for delta in range(len(deltas)):
        
        # Default parameters
        default_params = {"time_horizon": hwk.TIME_HORIZON}

        # Initialized parameters
        dict_args = {k: getattr(args, k, v) for k, v in default_params.items()}

        # Hawkes process hyper-parameters generation
        params, alpha, beta, _, mu = hyper_params_simulation(record=False)

        # Hawkes processes simulations
        simulated_events_seqs = hawkes_simulations(alpha, beta, mu, record=False)

        # Computed bins number
        num_bins = int(dict_args['time_horizon'] // delta)

        # Initialized array with dimensions (number of processes, number of jumps per unit of time)
        counts = np.zeros((len(simulated_events_seqs), num_bins), dtype=np.float32)

        # For each process (j), compute jump times histogram (h) using intervals boundaries specified by bins
        for j, h in enumerate(simulated_events_seqs):
            counts[j], _ = np.histogram(h, bins=np.linspace(0, dict_args['time_horizon'], num_bins + 1))

        # Written parquet file
        write_parquet(pl.DataFrame(counts, schema=np.arange(dict_args['time_horizon'], dtype=np.int32).astype(str).tolist()), filename=f"binned_hawkes_simulations_delta_{delta}.parquet")

        # Loaded model and predictions
        model = MLPTrainer()
        print(model.load_model())
        val_y_pred, _, _ = model.predict(val_x)

        # Computed errors
        errors = np.array([val_y_pred[i] - params[i] for i in range(number_of_tests)], dtype=np.float32)
        #outlier_mask = (errors > np.quantile(errors, 0.75) + iqr(errors) * 1.5) | (errors < np.quantile(errors, 0.25) - iqr(errors) * 1.5)
    
    # eta_error_data = np.zeros((number_of_tests, 3))
    # mu_error_data = np.zeros((number_of_tests, 3))

    # eta_error_data[expected_activity == activity, :3] = np.where(outlier_mask, 1, 0)
    # mu_error_data[expected_activity == activity, :3] = np.where(outlier_mask, 1, 0)

    # Built boxplots
    plt.style.use(['science', 'ieee'])

    _, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(18, 9))

    for ax, error_data, title in zip([ax1, ax2], [eta_error_data, mu_error_data], ["Effect on $\eta$ estimation", "Effect on $\mu$ estimation"]):

        ax.grid(which='major', color='#999999', linestyle='--')
        ax.minorticks_on()
        ax.grid(which='minor', color='#999999', linestyle='--', alpha=0.25)

        ax.boxplot([error_data[error_data['Expected_Activity']== activity]['Error'] for activity in activities], positions=activities, showfliers=False)
        
        ax.set_title(title, fontsize=16, pad=15)
        ax.set_xlabel('Expected Activity', fontsize=16, labelpad=15)
        ax.set_ylabel('Error', fontsize=16, labelpad=15)
        ax.tick_params(axis='both', which='major', labelsize=14, pad=8)
        #ax.scatter(error_data[error_data['outlier']==True]['Expected_Activity'], error_data[error_data['outlier']==True]['Error'], alpha=0.6)
        

    plt.savefig(os.path.join(dict_args['dirpath'], folder, filename), backend='pgf')
    plt.show()