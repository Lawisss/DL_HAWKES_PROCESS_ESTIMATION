#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Expected Activity effect module

File containing expected activity effect function

"""

import numpy as np

activity_ranges = [50, 100, 250, 500, 1000]
results = []

np.random.seed(0)

for i in range(len(activity_ranges)):
    print(i+1)
    
    min_eta = 0.2
    max_eta = 0.6
    min_beta = 1
    max_beta = 3
    expected_activity = activity_ranges[i]
    horizon = 100
    discretise_step = 1
    number_of_tests = 100
    number_of_test_processes = 200
    
    test_average_events = np.random.normal(expected_activity, 10, number_of_tests)
    
    test_avg_events_per_time = test_average_events / horizon
    test_eta = np.random.uniform(min_eta, max_eta, number_of_tests)
    test_mu = test_avg_events_per_time * (1 - test_eta)
    
    test_beta = np.random.uniform(min_beta, max_beta, number_of_tests)
    test_alpha = test_beta * test_eta
    test_params = np.column_stack((test_alpha, test_beta, test_mu))
    
    training = eta_mu_model(100000, horizon, expected_activity, discretise_step, 
                            min_eta=min_eta, max_eta=max_eta, min_beta=min_beta, max_beta=max_beta)
    
    raw_results = []
    errors = []
    
    for k in range(number_of_tests):
        raw_test_processes = []
        for j in range(number_of_test_processes):
            raw_test_processes.append(simulateHawkes(test_mu[k], test_alpha[k], test_beta[k], horizon))
        test_processes = discretise(raw_test_processes, discretise_step, horizon)
        raw_results.append(predict_hawkes_parameters(training['model'], test_processes, 
                                                     training['training_processes'], training['training_params']))
        errors.append(np.column_stack((raw_results[k]['alpha_est'], raw_results[k]['beta_est'], raw_results[k]['mu_est'])) - test_params[k])
    
    results.append({'raw_results': np.vstack(raw_results), 'errors': np.vstack(errors)})
