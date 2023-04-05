#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Hawkes module

File containing Hawkes Process function (simulation/estimation).

"""

import numpy as np
import pandas as pd
import Hawkes as hk

from UTILS.utils import write_csv

def hawkes_simulation(kernel='exp', baseline='const', params={"mu": 0.1, "alpha": 0.5, "beta": 10.0}, interval=[0,100]):

    hawkes_process = hk.simulator().set_kernel(kernel).set_baseline(baseline).set_parameter(params)
    T = hawkes_process.simulate(interval)
    
    hawkes_process.plot_N()
    hawkes_process.plot_l()

    return hawkes_process, T


def hawkes_estimation(T, kernel='exp', baseline='const', interval=[0,100], end_T=200, num_seq=100, filepath="C:/Users/Nicolas Girard/Documents/VAE_HAWKES_PROCESS_ESTIMATION/CODE/RESULTS/hawkes_estimation.csv"):
    
    hawkes_process = hk.estimator().set_kernel(kernel).set_baseline(baseline)
    hawkes_process.fit(T, interval)
    
    metrics = {'Event(s)': len(T),
               'Parameters': {k: round(v, 3) for k, v in hawkes_process.para.items()},
               'Branching Ratio': round(hawkes_process.br, 3),
               'Log-Likelihood': round(hawkes_process.L, 3),
               'AIC': round(hawkes_process.AIC, 3)}
    
    write_csv(metrics, filepath)

    [T_transform, interval_transform] = hawkes_process.t_trans() 
    hawkes_process.plot_KS()

    T_pred = hawkes_process.predict(end_T, num_seq) 
    hawkes_process.plot_N_pred()   
    
    return T_pred, metrics, T_transform, interval_transform