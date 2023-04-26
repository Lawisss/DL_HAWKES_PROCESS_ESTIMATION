#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Hyper-parameters MPI module

File containing parallelized Hawkes process hyper-parameters generation functions (Default Parameter Values)

"""

from typing import Tuple, Optional, Callable

import numpy as np
from mpi4py import MPI

import VARIABLES.hawkes_var as hwk
from UTILS.utils import write_parquet

# Parallelized generated Hawkes process hyper-parameters (alpha, beta, mu)

def hyper_params_simulation(root: int = 0, filename: str = "hawkes_hyperparams.parquet", args: Optional[Callable] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    """
    Generated and saved Hawkes process hyperparameters

    Args:
        filename (str, optional): Filename to save hyperparameters in CSV file (default: "hawkes_hyperparams.parquet")
        root (int, optional): Rank of process to use as root for MPI communications (default: 0)
        args (Callable, optional): Arguments if you use main.py instead of tutorial.ipynb

    Returns:
        A tuple containing:
        - Alpha, beta, and mu parameters for each process
        - Alpha parameters for each process
        - Beta parameters for each process
        - Mu parameters for each process
    """
    
    # Initialized MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Default parameters
    default_params = {"expected_activity": hwk.EXPECTED_ACTIVITY,
                      "std": hwk.STD,
                      "process_num": hwk.PROCESS_NUM,
                      "min_itv_eta": hwk.MIN_ITV_ETA,
                      "max_itv_eta": hwk.MAX_ITV_ETA,
                      "min_itv_beta": hwk.MIN_ITV_BETA,
                      "max_itv_beta": hwk.MAX_ITV_BETA,
                      "time_horizon": hwk.TIME_HORIZON}

    # Initialized parameters
    dict_args = {k: getattr(args, k, v) for k, v in default_params.items()}

    # Generated random vectors on root process
    if rank == 0:
        epsilon = np.random.normal(dict_args['expected_activity'], dict_args['std'], dict_args['process_num'])
        eta = np.random.uniform(dict_args['min_itv_eta'], dict_args['max_itv_eta'], dict_args['process_num'])
        beta = np.random.uniform(dict_args['min_itv_beta'], dict_args['max_itv_beta'], dict_args['process_num'])

    # Broadcast random vectors to all processes
    epsilon = comm.bcast(epsilon, root=root)
    eta = comm.bcast(eta, root=root)
    beta = comm.bcast(beta, root=root)

    # Divided vectors indices among processes
    indices = np.array_split(range(dict_args['process_num']), size)

    # Scattered indices to all processes
    indices = comm.scatter(indices, root=root)

    # Calculated alpha/mu vectors in parallel
    alpha = np.zeros(dict_args['process_num'], dtype=np.float32)
    mu = np.zeros(dict_args['process_num'], dtype=np.float32)

    alpha[indices] = eta[indices]
    mu[indices] = (epsilon[indices] / dict_args['time_horizon']) * (1 - eta[indices])
    
    # Reduced alpha/mu vectors from all processes to root process
    comm.Reduce(alpha, np.zeros(dict_args['process_num'], dtype=np.float32), op=MPI.SUM, root=root)
    comm.Reduce(mu, np.zeros(dict_args['process_num'], dtype=np.float32), op=MPI.SUM, root=root)

    # Written parameters to Parquet file
    if rank == 0:
        write_parquet({"alpha": alpha, "beta": beta, "mu": mu}, filename=filename)
        
        # Written CSV file on the root process
        # params = [{"alpha": a, "beta": b, "mu": m} for a, b, m in zip(alpha, beta, mu)]
        # write_csv(params, filename=filename)

        return np.array([alpha, beta, mu], dtype=np.float32).T, alpha, beta, mu