# -*- coding: utf-8 -*-

"""Variable module

File containing all project variables.

"""

import os
import socket
from datetime import datetime

import torch
from torch.profiler import ProfilerActivity


# Hawkes process hyper-parameters generation parameters (hyperparameters.py)

MIN_ITV_BETA = 1                                                       # Beta minimum interval
MAX_ITV_BETA = 3                                                       # Beta maximum interval

MIN_ITV_ETA = 0.05                                                     # Eta minimum interval
MAX_ITV_ETA = 0.8                                                      # Eta maximum interval

EXPECTED_ACTIVITY = 500                                                # Total number of expected events
STD = 50                                                               # Standard deviation for generating epsilon

# Hawkes Process simulation/estimation parameters (hawkes.py, discretisation.py)

KERNEL = 'exp'                                                         # Type of kernel function
BASELINE = 'const'                                                     # Type of baseline function

TIME_ITV_START = 0                                                     # Start time interval for simulation
TIME_HORIZON = 100                                                     # Time horizon for simulation

PROCESS_NUM = 50                                                       # Number of processes to simulate

END_T = 200                                                            # End time for estimation
NUM_SEQ = 100                                                          # Number of sequences for estimation

# Discretisation parameters (discretisation.py)

DISCRETISE_STEP = 1                                                    # Discretise step = Delta

# Datasets parameters (dataset.py)

VAL_RATIO = 0.25                                                       # Fraction of data used for validation
TEST_RATIO = 0.125                                                     # Fraction of data used for testing
BATCH_SIZE = 128                                                       # Number of samples used in each process iteration
SHUFFLE = True                                                         # Shuffle data in each epoch
DROP_LAST = True                                                       # Drop last inchoate batch if batch size ∤ dataset size
NUM_WORKERS = 4                                                        # Number of worker processes to use for data loading
PIN_MEMORY = True                                                      # Copy tensors to pinned memory

# MLP parameters (mlp.py)

INPUT_SIZE = 100_000                                                   # MLP input size
HIDDEN_SIZE = 100                                                      # Number of neurons in hidden layers
OUTPUT_SIZE = 2                                                        # MLP output size
NUM_HIDDEN_LAYERS = 6                                                  # MLP number of hidden layers
L2_REG = 0.001                                                         # L2 regularization parameter

LEARNING_RATE = 0.01                                                   # Learning rate for optimizer
MAX_EPOCHS = 500                                                       # Maximum number of epochs for training

FILENAME_BEST_MODEL = "best_model.pt"                                  # Filename to save best model weights
EARLY_STOP_PATIENCE = 25                                               # Epochs without improvement before early stopping
EARLY_STOP_DELTA = 0.01                                                # Minimum reduction of val_loss to consider improvement

# Model summary (mlp.py)

SUMMARY_MODEL = "MLP"                                                  # Model name used for summary
SUMMARY_MODE = "train"                                                 # Summary modes: "train"/"eval"
SUMMARY_VERBOSE = 2                                                    # Level of verbosity in summary
SUMMARY_COL_NAMES = ("input_size", "output_size", 
                     "num_params", "params_percent",                   # Columns names used in summary
                     "kernel_size", "mult_adds", "trainable")  


# Tensorboard (Metrics evaluation - mlp.py)

LOGDIRUN = os.path.abspath("RESULTS/RUNS")                             # Tensorboard logs directory for each run
RUN_NAME = datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + "_" \
           + socket.gethostname().upper()                              # Name for current run based on timestamp/hostname

# Tensorboard (Fonction profiling - utils.py)

LOGDIPROF = os.path.abspath("RESULTS/PROFILING")                       # Profiling results directory 

ACTIVITIES = [ProfilerActivity.CPU, ProfilerActivity.CUDA]             # CPU and CUDA profiling
WAIT = 1                                                               # Time (in seconds) to wait before starting profiling
WARMUP = 1                                                             # Time (in seconds) for warming up before profiling
ACTIVE = 2                                                             # Time (in seconds) for profiling
REPEAT = 0                                                             # Number of times to repeat profiling
SKIP_FIRST = 0                                                         # Number of first profiling results to discard
RECORD_SHAPES = True                                                   # Record tensor shapes in profiling output
PROFILE_MEMORY = False                                                 # Include memory profiling
WITH_STACK = False                                                     # Include function call stack in profiling output
WITH_FLOPS = False                                                     # Include FLOPS computation in profiling output
WITH_MODULES = False                                                   # Include profiling of module operations

GROUP_BY_INPUT_SHAPE = False                                           # Group profiling output by tensor input shapes
GROUP_BY_STACK_N = 0                                                   # Stack frames number to include in function call stack
SORT_BY = "cpu_time_total"                                             # Sort profiling output by specified metric
ROW_LIMIT = 10                                                         # Maximum number of rows to display in profiling output

# Device parameter (dataset.py, mlp.py, utils.py)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Check if GPU is available, otherwise use CPU

# Global filepath to store results from modules (utils.py)

FILEPATH = os.path.abspath("RESULTS")                                  # Absolute path to results directory         