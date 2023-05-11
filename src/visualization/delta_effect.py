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
from preprocessing.dataset import split_data
from evaluation.eval import compute_errors
from hawkes.simulation import hawkes_simulations
from hawkes.hyperparameters import hyper_params_simulation


def delta_generation(deltas: List = [0.25, 0.5, 1.0, 2.0, 5.0], number_of_tests: int = 2, folder: str = "photos", filename: str = "activity_effect.pdf", args: Optional[Callable] = None):

    # Initialized lists
    counts = []
    errors = []

    # Loaded model
    model = MLPTrainer()
    print(model.load_model())

    for delta in deltas:
        
        # Default parameters
        default_params = {"time_horizon": hwk.TIME_HORIZON}

        # Initialized parameters
        dict_args = {k: getattr(args, k, v) for k, v in default_params.items()}

        # Hawkes process hyper-parameters generation
        params, alpha, beta, eta, mu = hyper_params_simulation(record=False)

        # Computed bins number
        num_bins = int(dict_args['time_horizon'] // delta)

        # Launched test
        for i in range(number_of_tests):
            
            # Hawkes processes simulations
            simulated_events_seqs = hawkes_simulations(alpha, beta, mu, record=False)

            # Initialized array with dimensions and type
            count = np.zeros((len(simulated_events_seqs), num_bins), dtype=np.float32)
            error = np.zeros((number_of_tests, num_bins), dtype=np.float32)
            
            # Discretisation
            for j, h in enumerate(simulated_events_seqs):
                count[j], _ = np.histogram(h, bins=np.linspace(0, dict_args['time_horizon'], num_bins + 1))

            # Predicted eta/mu
            test_x, test_y, _, _, _, _ = split_data(count, np.column_stack((eta, mu)))
            y_pred, _, _ = model.predict(test_x)
            print(y_pred)
            # Computed errors
            error[i] = compute_errors(test_y, y_pred, model_name='MLP')

        counts.append(count)
        errors.append(error)

        # Written parquet file
        #write_parquet(pl.DataFrame({"binned_simulations": counts}, filename=f"binned_hawkes_simulations_delta_{delta}.parquet"))

    return counts, errors


    for k in range(number_of_tests):
        raw_test_processes = [simulate_hawkes(test_mu[k], test_alpha[k], test_beta[k], horizon)
                              for j in range(number_of_test_processes)]
        test_processes = discretise(raw_test_processes, discretise_step, horizon)
        raw_results.append(predict_hawkes_parameters(training["model"], test_processes,
                                                     training["training_processes"], training["training_params"]))
        errors.append(np.column_stack((raw_results[k]["alpha_est"], raw_results[k]["beta_est"], raw_results[k]["mu_est"])) - test_params[k])

    results.append({"raw_results": np.array(raw_results).reshape(-1, 4), "errors": np.array(errors).reshape(-1, 3)})
        


    #     # Computed errors
    #     errors = np.array([val_y_pred[i] - params[i] for i in range(number_of_tests)], dtype=np.float32)
    #     #outlier_mask = (errors > np.quantile(errors, 0.75) + iqr(errors) * 1.5) | (errors < np.quantile(errors, 0.25) - iqr(errors) * 1.5)
    
    # # eta_error_data = np.zeros((number_of_tests, 3))
    # # mu_error_data = np.zeros((number_of_tests, 3))

    # # eta_error_data[expected_activity == activity, :3] = np.where(outlier_mask, 1, 0)
    # # mu_error_data[expected_activity == activity, :3] = np.where(outlier_mask, 1, 0)

    # # Built boxplots
    # plt.style.use(['science', 'ieee'])

    # _, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(18, 9))

    # for ax, error_data, title in zip([ax1, ax2], [eta_error_data, mu_error_data], ["Effect on $\eta$ estimation", "Effect on $\mu$ estimation"]):

    #     ax.grid(which='major', color='#999999', linestyle='--')
    #     ax.minorticks_on()
    #     ax.grid(which='minor', color='#999999', linestyle='--', alpha=0.25)

    #     ax.boxplot([error_data[error_data['Expected_Activity']== activity]['Error'] for activity in activities], positions=activities, showfliers=False)
        
    #     ax.set_title(title, fontsize=16, pad=15)
    #     ax.set_xlabel('Expected Activity', fontsize=16, labelpad=15)
    #     ax.set_ylabel('Error', fontsize=16, labelpad=15)
    #     ax.tick_params(axis='both', which='major', labelsize=14, pad=8)
    #     #ax.scatter(error_data[error_data['outlier']==True]['Expected_Activity'], error_data[error_data['outlier']==True]['Error'], alpha=0.6)
        

    # plt.savefig(os.path.join(dict_args['dirpath'], folder, filename), backend='pgf')
    # plt.show()