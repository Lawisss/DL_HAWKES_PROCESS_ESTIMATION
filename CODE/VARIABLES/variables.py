# -*- coding: utf-8 -*-

"""Variable module

File containing all project variables.

"""

# Hawkes Process simulation/estimation parameters (hawkes.py, discretisation.py)
KERNEL = 'exp'
BASELINE = 'const'

TIME_ITV_START = 0
TIME_HORIZON = 100

TRAINING_PROCESS = 100_000
TESTING_PROCESS = 0

# Especially used in the hawkes_estimation function
END_T = 200
NUM_SEQ = 100

# Hawkes process hyper-parameters generation parameters (hyperparameters.py)
MIN_ITV_BETA = 1 
MAX_ITV_BETA = 3

MIN_ITV_ETA = 0.05
MAX_ITV_ETA = 0.8

EXPECTED_ACTIVITY = 500
STD = 50

# Discretisation parameters (discretisation.py)
# Discretise step = Delta
DISCRETISE_STEP = 1

# Global filepath to store different results from different modules (hawkes.py, hyperparameters.py)
FILEPATH = "C:/Users/Nicolas Girard/Documents/VAE_HAWKES_PROCESS_ESTIMATION/CODE/RESULTS/"