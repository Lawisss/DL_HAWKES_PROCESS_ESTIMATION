#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Hyper-parameters MPI module

File containing parallelized Hawkes process hyper-parameters generation functions (Default Parameter Values).

"""

import numpy as np
from mpi4py import MPI
from typing import Tuple

import VARIABLES.variables as var
from UTILS.utils import write_csv

# Parallelized generated Hawkes process hyper-parameters (alpha, beta, mu)

def hyper_params_simulation(filename: str = "hawkes_hyperparams.csv") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    # Initialized MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Generated random vectors on root process
    if rank == 0:
        epsilon = np.random.normal(var.EXPECTED_ACTIVITY, var.STD, var.PROCESS_NUM)
        eta = np.random.uniform(var.MIN_ITV_ETA, var.MAX_ITV_ETA, var.PROCESS_NUM)
        beta = np.random.uniform(var.MIN_ITV_BETA, var.MAX_ITV_BETA, var.PROCESS_NUM)

    # Broadcast random vectors to all processes
    epsilon = comm.bcast(epsilon, root=0)
    eta = comm.bcast(eta, root=0)
    beta = comm.bcast(beta, root=0)

    # Divided vectors indices among processes
    indices = np.array_split(range(var.PROCESS_NUM), size)

    # Scattered indices to all processes
    indices = comm.scatter(indices, root=0)

    # Calculated alpha / mu vectors in parallel
    alpha = np.zeros(var.PROCESS_NUM, dtype=np.float64)
    mu = np.zeros(var.PROCESS_NUM, dtype=np.float64)

    alpha[indices] = eta[indices]
    mu[indices] = (epsilon[indices] / var.TIME_HORIZON) * (1 - eta[indices])
    
    # Reduced alpha / mu vectors from all processes to root process
    comm.Reduce(alpha, None, op=MPI.SUM, root=0)
    comm.Reduce(mu, None, op=MPI.SUM, root=0)

    # Written parameters to a CSV file on the root process
    if rank == 0:
        params = [{"alpha": a, "beta": b, "mu": m} for a, b, m in zip(alpha, beta, mu)]
        write_csv(params, filepath=f"{var.FILEPATH}{filename}")

        return np.array([alpha, beta, mu], dtype=np.float64).T, alpha, beta, mu