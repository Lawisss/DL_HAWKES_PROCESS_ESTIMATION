# -*- coding: utf-8 -*-

"""Variable module

File containing all project variables.

"""

import os
import socket
from datetime import datetime

import torch

# Hawkes process hyper-parameters generation parameters (hyperparameters.py)
MIN_ITV_BETA = 1 
MAX_ITV_BETA = 3

MIN_ITV_ETA = 0.05
MAX_ITV_ETA = 0.8

EXPECTED_ACTIVITY = 500
STD = 50

# Hawkes Process simulation/estimation parameters (hawkes.py, discretisation.py)
KERNEL = 'exp'
BASELINE = 'const'

TIME_ITV_START = 0
TIME_HORIZON = 100

PROCESS_NUM = 50

# Especially used in the hawkes_estimation function
END_T = 200
NUM_SEQ = 100

# Discretisation parameters (discretisation.py)
DISCRETISE_STEP = 1 # Discretise step = Delta

# Datasets parameters (dataset.py)
VAL_RATIO = 0.25 # Fraction of data used for validation
TEST_RATIO = 0.125 # Fraction of data used for testing
BATCH_SIZE = 128
SHUFFLE = True # Shuffle data in each epoch
DROP_LAST = True # Drop last incomplete batch if dataset size not divisible by batch size
NUM_WORKERS = 4 # Number of worker processes to use for data loading
PIN_MEMORY = True # Copy the tensors to pinned memory

# MLP parameters (mlp.py)
INPUT_SIZE = 100_000
HIDDEN_SIZE = 100
OUTPUT_SIZE = 2
NUM_HIDDEN_LAYERS = 6
L2_REG = 0.001

LEARNING_RATE = 0.01
MAX_EPOCHS = 500

FILENAME_BEST_MODEL = "best_model.pt"
EARLY_STOP_PATIENCE = 25 # Number of epochs without improvement before triggering early stopping
EARLY_STOP_DELTA = 0.01 # Minimum reduction in validation loss to consider an improvement

# Tensorboard
LOGDIR = os.path.abspath("RESULTS/RUNS")
RUN_NAME = datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + "_" + socket.gethostname().upper()

# Model summary
SUMMARY_MODEL = "MLP"
SUMMARY_COL_NAMES = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable")
SUMMARY_MODE = "train" # Two modes: train/eval
SUMMARY_VERBOSE = 2

# Device parameter (dataset.py, mlp.py)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global filepath to store different results from different modules (hawkes.py, hyperparameters.py)
FILEPATH = os.path.abspath("RESULTS")