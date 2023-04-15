#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Hyper-parameters MPI module

File containing parallelized Hawkes process hyper-parameters generation functions (Default Parameter Values).

"""

from typing import Tuple

import numpy as np
from mpi4py import MPI

import VARIABLES.hawkes_var as hwk
from UTILS.utils import write_csv

# Parallelized generated Hawkes process hyper-parameters (alpha, beta, mu)

def hyper_params_simulation(root: int = 0, filename: str = "hawkes_hyperparams.csv") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    """_summary_

    Returns:
        _type_: _description_

    """
    
    # Initialized MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Generated random vectors on root process
    if rank == 0:
        epsilon = np.random.normal(hwk.EXPECTED_ACTIVITY, hwk.STD, hwk.PROCESS_NUM)
        eta = np.random.uniform(hwk.MIN_ITV_ETA, hwk.MAX_ITV_ETA, hwk.PROCESS_NUM)
        beta = np.random.uniform(hwk.MIN_ITV_BETA, hwk.MAX_ITV_BETA, hwk.PROCESS_NUM)

    # Broadcast random vectors to all processes
    epsilon = comm.bcast(epsilon, root=root)
    eta = comm.bcast(eta, root=root)
    beta = comm.bcast(beta, root=root)

    # Divided vectors indices among processes
    indices = np.array_split(range(hwk.PROCESS_NUM), size)

    # Scattered indices to all processes
    indices = comm.scatter(indices, root=root)

    # Calculated alpha/mu vectors in parallel
    alpha = np.zeros(hwk.PROCESS_NUM, dtype=np.float32)
    mu = np.zeros(hwk.PROCESS_NUM, dtype=np.float32)

    alpha[indices] = eta[indices]
    mu[indices] = (epsilon[indices] / hwk.TIME_HORIZON) * (1 - eta[indices])
    
    # Reduced alpha/mu vectors from all processes to root process
    comm.Reduce(alpha, np.zeros(hwk.PROCESS_NUM, dtype=np.float32), op=MPI.SUM, root=root)
    comm.Reduce(mu, np.zeros(hwk.PROCESS_NUM, dtype=np.float32), op=MPI.SUM, root=root)

    # Written CSV file on the root process
    if rank == 0:
        params = [{"alpha": a, "beta": b, "mu": m} for a, b, m in zip(alpha, beta, mu)]
        write_csv(params, filename=filename)

        return np.array([alpha, beta, mu], dtype=np.float32).T, alpha, beta, mu