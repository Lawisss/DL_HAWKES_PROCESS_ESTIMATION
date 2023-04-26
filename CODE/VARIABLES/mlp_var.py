# -*- coding: utf-8 -*-

"""MLP variables module

File containing all MLP variables.

"""

from typing import Tuple

from main import args

# MLP parameters (mlp.py)

INPUT_SIZE: int = 100 if not args.input_size else args.input_size                                        # MLP input size
HIDDEN_SIZE: int = 100 if not args.hidden_size else args.hidden_size                                     # Number of neurons in hidden layers
OUTPUT_SIZE: int = 2 if not args.output_size else args.output_size                                       # MLP output size
NUM_HIDDEN_LAYERS: int = 6 if not args.num_hidden_layers else args.num_hidden_layers                     # MLP number of hidden layers
L2_REG: float = 0.001 if not args.l2_reg else args.l2_reg                                                # L2 regularization parameter

LEARNING_RATE: float = 0.01 if not args.learning_rate else args.learning_rate                            # Learning rate for optimizer
MAX_EPOCHS: int = 500 if not args.max_epochs else args.max_epochs                                        # Maximum number of epochs for training

FILENAME_BEST_MODEL: str = "best_model.pt" if not args.filename_best_model else args.filename_best_model # Filename to save best model weights
EARLY_STOP_PATIENCE: int = 25 if not args.early_stop_patience else args.early_stop_patience              # Epochs without improvement before early stopping
EARLY_STOP_DELTA: float = 0.01 if not args.early_stop_delta else args.early_stop_delta                   # Minimum reduction of val_loss to consider improvement

# Model summary (mlp.py)

SUMMARY_MODEL: str = "MLP" if not args.summary_model else args.summary_model                             # Model name used for summary
SUMMARY_MODE: str = "train" if not args.summary_mode else args.summary_mode                              # Summary modes: "train"/"eval"
SUMMARY_VERBOSE: int = 1 if not args.summary_verbose else args.summary_verbose                           # Level of verbosity in summary
SUMMARY_COL_NAMES: Tuple[str, str, str, str, str, str, str] = \
                   ("input_size", "output_size", "num_params", 
                    "params_percent", "kernel_size", 
                    "mult_adds", "trainable") \
                    if not args.summary_col_names else args.summary_col_names                            # Columns names used in summary