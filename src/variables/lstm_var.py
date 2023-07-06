# -*- coding: utf-8 -*-

"""LSTM variables module

File containing all LSTM variables.

"""

import socket
from typing import Tuple
from datetime import datetime

import variables.hawkes_var as hwk

# LSTM parameters (mlp_model.py)

INPUT_SIZE: int = int(hwk.TIME_HORIZON // hwk.DISCRETISE_STEP)            # LSTM input size (HORIZON = 100)
HIDDEN_SIZE: int = int(hwk.TIME_HORIZON // hwk.DISCRETISE_STEP)           # Neurons in hidden layers (HORIZON = 100)
OUTPUT_SIZE: int = 2                                                      # LSTM output size
NUM_HIDDEN_LAYERS: int = 6                                                # LSTM number of hidden layers
L2_REG: float = 0.001                                                     # L2 regularization parameter

LEARNING_RATE: float = 0.001                                              # Learning rate for optimizer
MAX_EPOCHS: int = 500                                                     # Maximum number of epochs for training

FILENAME_BEST_MODEL: str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S") \
            + "_" + socket.gethostname().upper() + "_" + "best_model.pt"  # Filename to save best model weights
EARLY_STOP_PATIENCE: int = 25                                             # Epochs without improvement before early stopping
EARLY_STOP_DELTA: float = 0.01                                            # Minimum reduction of val_loss to consider improvement

# Model summary (mlp_model.py)

SUMMARY_MODEL: str = "LSTM"                                               # Model name used for summary
SUMMARY_MODE: str = "train"                                               # Summary modes: "train"/"eval"
SUMMARY_VERBOSE: int = 1                                                  # Level of verbosity in summary
SUMMARY_COL_NAMES: Tuple[str, str, str, str, str, str, str] = \
                   ("input_size", "output_size", "num_params", 
                    "params_percent", "kernel_size", 
                    "mult_adds", "trainable")                             # Columns names used in summary