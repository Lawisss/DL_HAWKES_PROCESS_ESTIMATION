#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Expected Activity effect module

File containing expected activity effect function

"""

import os
from typing import List, Union, Callable, Optional

import numpy as np
from scipy.stats import iqr
import matplotlib.pyplot as plt

import variables.hawkes_var as hwk
from dl.mlp_model import MLPTrainer

def activity_effect(val_x, activities: List = [50, 100, 250, 500, 1000], number_of_tests: int = 100, folder: str = "photos", filename: str = "activity_effect.pdf", args: Optional[Callable] = None):

    for activity in range(len(activities)):

        # Default parameters
        default_params = {"std": hwk.STD,
                          "process_num": hwk.PROCESS_NUM,
                          "min_itv_eta": hwk.MIN_ITV_ETA,
                          "max_itv_eta": hwk.MAX_ITV_ETA,
                          "min_itv_beta": hwk.MIN_ITV_BETA,
                          "max_itv_beta": hwk.MAX_ITV_BETA,
                          "time_horizon": hwk.TIME_HORIZON}

        # Initialized parameters
        dict_args = {k: getattr(args, k, v) for k, v in default_params.items()}

        # Generated random vectors of size PROCESS_NUM (epsilon = average of events)
        epsilon = np.random.normal(activity, dict_args['std'], dict_args['process_num'])
        eta = np.random.uniform(dict_args['min_itv_eta'], dict_args['max_itv_eta'], dict_args['process_num'])

        # Calculated alpha/mu vectors from beta/eta vectors (alpha = eta because of library exponential formula)
        alpha = eta
        mu = (epsilon / dict_args['time_horizon']) * (1 - eta)
        params = np.column_stack((alpha, mu)).astype(np.float32)

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

       