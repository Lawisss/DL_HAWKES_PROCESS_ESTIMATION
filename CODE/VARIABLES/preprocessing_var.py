# -*- coding: utf-8 -*-

"""Preprocessing variables module

File containing all preprocessing variables.

"""

import os

import torch

from main import args

# Datasets parameters (dataset.py)

VAL_RATIO: float = 0.25 if not args.val_ratio else args.val_ratio               # Fraction of data used for validation
TEST_RATIO: float = 0.125 if not args.test_ratio else args.test_ratio           # Fraction of data used for testing
BATCH_SIZE: int = 128 if not args.batch_size else args.batch_size               # Number of samples used in each process iteration
SHUFFLE: bool = True if not args.shuffle else args.shuffle                      # Shuffle data in each epoch
DROP_LAST: bool = True if not args.drop_last else args.drop_last                # Drop last inchoate batch if batch size ∤ dataset size
NUM_WORKERS: int = 4 if not args.num_workers else args.num_workers              # Number of worker processes to use for data loading
PIN_MEMORY: bool = True if not args.pin_memory else args.pin_memory             # Copy tensors to pinned memory

# Device parameter (dataset.py, mlp.py, utils.py)

DEVICE: torch.device = \
        torch.device("cuda" if torch.cuda.is_available() else "cpu") \
        if not args.device else args.device                                     # Check if GPU is available, otherwise use CPU

# Writing/Reading parameters (utils.py)

DIRPATH: str = os.path.abspath("RESULTS") if not args.dirpath else args.dirpath # Absolute path to results directory  
