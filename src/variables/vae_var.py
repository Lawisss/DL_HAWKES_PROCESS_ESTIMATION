# -*- coding: utf-8 -*-

"""VAE variables module

File containing all VAE variables.

"""

import socket
from typing import Tuple
from datetime import datetime

import variables.hawkes_var as hwk

# VAE parameters (vae_model.py, dueling_decoder.py)

INPUT_SIZE: int = int(hwk.TIME_HORIZON // hwk.DISCRETISE_STEP)            # VAE input size (HORIZON = 100)
LATENT_SIZE: int = 15                                                     # Latent space size (HORIZON = 100)
INTERMEDIATE_SIZE: int = int(INPUT_SIZE * 0.75)                           # VAE output size
LEARNING_RATE: float = 0.001                                              # Learning rate for optimizer
MAX_EPOCHS: int = 10_000                                                  # Maximum number of epochs for training

FILENAME_BEST_MODEL: str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S") \
            + "_" + socket.gethostname().upper() + "_" + "best_model.pt"  # Filename to save best model weights

KL_START: int = 2000                                                      # KL divergence initial value
KL_STEEP: int = 1000                                                      # KL divergence increase factor
ANNEAL_TARGET: int = 1                                                    # KL divergence target value
MIN_CYCLES: int = 8                                                       # Minimum number of training cycles

# Model summary (vae_model.py)

SUMMARY_MODEL: str = "VAE"                                                # Model name used for summary
SUMMARY_MODE: str = "train"                                               # Summary modes: "train"/"eval"
SUMMARY_VERBOSE: int = 1                                                  # Level of verbosity in summary
SUMMARY_COL_NAMES: Tuple[str, str, str, str, str, str, str] = \
                   ("input_size", "output_size", "num_params", 
                    "params_percent", "kernel_size", 
                    "mult_adds", "trainable")                             # Columns names used in summary