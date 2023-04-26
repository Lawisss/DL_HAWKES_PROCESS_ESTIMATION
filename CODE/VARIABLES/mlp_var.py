# -*- coding: utf-8 -*-

"""MLP variables module

File containing all MLP variables.

"""

from typing import Tuple

# MLP parameters (mlp.py)

INPUT_SIZE: int = 100                                               # MLP input size
HIDDEN_SIZE: int = 100                                              # Number of neurons in hidden layers
OUTPUT_SIZE: int = 2                                                # MLP output size
NUM_HIDDEN_LAYERS: int = 6                                          # MLP number of hidden layers
L2_REG: float = 0.001                                               # L2 regularization parameter

LEARNING_RATE: float = 0.01                                         # Learning rate for optimizer
MAX_EPOCHS: int = 500                                               # Maximum number of epochs for training

FILENAME_BEST_MODEL: str = "best_model.pt"                          # Filename to save best model weights
EARLY_STOP_PATIENCE: int = 25                                       # Epochs without improvement before early stopping
EARLY_STOP_DELTA: float = 0.01                                      # Minimum reduction of val_loss to consider improvement

# Model summary (mlp.py)

SUMMARY_MODEL: str = "MLP"                                          # Model name used for summary
SUMMARY_MODE: str = "train"                                         # Summary modes: "train"/"eval"
SUMMARY_VERBOSE: int = 1                                            # Level of verbosity in summary
SUMMARY_COL_NAMES: Tuple[str, str, str, str, str, str, str] = \
                   ("input_size", "output_size", "num_params", 
                    "params_percent", "kernel_size", 
                    "mult_adds", "trainable")                       # Columns names used in summary