# -*- coding: utf-8 -*-

"""Preprocessing variables module

File containing all preprocessing variables.

"""

import os

import torch

import variables.hawkes_var as hwk

# Datasets parameters (dataset.py)

VAL_RATIO: float = 0.00                                               # Fraction of data used for validation
TEST_RATIO: float = 0.00                                              # Fraction of data used for testing
BATCH_SIZE: int = int(hwk.PROCESS_NUM * 0.1)                          # Number of samples used in each process iteration
SHUFFLE: bool = True                                                  # Shuffle data in each epoch
DROP_LAST: bool = True                                                # Drop last inchoate batch if batch size ∤ dataset size
NUM_WORKERS: int = 4                                                  # Number of worker processes to use for data loading
PIN_MEMORY: bool = True                                               # Copy tensors to pinned memory

# Device parameter (dataset.py, mlp.py, utils.py)

DEVICE: torch.device = \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Check if GPU is available, otherwise use CPU

# Writing/Reading parameters (utils.py)

DIRPATH: str = os.path.abspath("results")                             # Absolute path to results directory
