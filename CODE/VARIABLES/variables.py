# -*- coding: utf-8 -*-

"""Variable module

File containing all project variables.

"""

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

# MLP parameters (mlp.py)
INPUT_SIZE = 100_000
BATCH_SIZE = 128
HIDDEN_SIZE = 100
OUTPUT_SIZE = 2
NUM_HIDDEN_LAYERS = 6
L2_REG = 0.001
MAX_EPOCHS = 500
EARLY_STOP_PATIENCE = 25 # Number of epochs without improvement before triggering early stopping
EARLY_STOP_DELTA = 0.01 # Minimum reduction in validation loss to consider an improvement
VAL_RATIO = 0.25 # Fraction of data used for validation
TEST_RATIO = 0.125 # Fraction of data used for testing

# Global filepath to store different results from different modules (hawkes.py, hyperparameters.py)
FILEPATH = "C:/Users/Nicolas Girard/Documents/VAE_HAWKES_PROCESS_ESTIMATION/CODE/RESULTS/"