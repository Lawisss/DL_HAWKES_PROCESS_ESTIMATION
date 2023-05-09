#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Expected Activity effect module

File containing expected activity effect function

"""

import numpy as np

import variables.hawkes_var as hwk
from dl.mlp_model import MLPTrainer
from hawkes.discretisation import discretise
from hawkes.simulation import hawkes_simulations


def activity_effect(val_x, activities=[50, 100, 250, 500, 1000], number_of_tests=100, args=None):

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

       