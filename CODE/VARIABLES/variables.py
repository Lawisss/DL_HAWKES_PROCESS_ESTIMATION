# -*- coding: utf-8 -*-

"""Variable module

File containing all project variables.

"""

# Hawkes Process parameters (hawkes.py)
KERNEL = 'exp'
BASELINE = 'const'

TIME_ITV_START = 0
TIME_HORIZON = 100
TRAINING_PROCESS = 100_000
TESTING_PROCESS = 0

# Datasets generation parameters (generation.py)
MIN_ITV_BETA = 1 
MAX_ITV_BETA = 3

MIN_ITV_ETA = 0.05
MAX_ITV_ETA = 0.8

EXPECTED_ACTIVITY = 500
STD = 50


DISCRETISE_STEP = 0.1



END_T = 200
NUM_SEQ = 100

# Global filepath to store different results from different modules (hawkes.py, generation.py)
FILEPATH = "C:/Users/Nicolas Girard/Documents/VAE_HAWKES_PROCESS_ESTIMATION/CODE/RESULTS/"