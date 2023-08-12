# -*- coding: utf-8 -*-

""" Hawkes variables module

File containing all hawkes variables.

"""

# Hawkes process hyperparameters generation parameters (hyperparameters.py)

MIN_ITV_BETA: float = 1.0                 # Beta minimum interval (Exponential - decay rate)
MAX_ITV_BETA: float = 3.0                 # Beta maximum interval (Exponential - decay rate)

MIN_ITV_K: float = 0.01                   # K minimum interval (Power law - amplitude ratio)
MAX_ITV_K: float = 10                     # K maximum interval (Power law - amplitude ratio)

MIN_ITV_C: float = 0.1                    # C minimum interval (Power law - scaling factor)
MAX_ITV_C: float = 2.0                    # C maximum interval (Power law - scaling factor)

MIN_ITV_P: float = 1.0                    # P minimum interval (Power law - power law exponent)
MAX_ITV_P: float = 3.0                    # P maximum interval (Power law - power law exponent)

MIN_ITV_ETA: float = 0.05                 # Eta minimum interval
MAX_ITV_ETA: float = 0.8                  # Eta maximum interval

EXPECTED_ACTIVITY: int = 500              # Total number of expected events 
STD: float = 10                           # Standard deviation for generating epsilon

# Hawkes Process simulation/estimation parameters (simulation.py, discretisation.py)

KERNEL: str = 'pow'                       # Type of kernel function
BASELINE: str = 'const'                   # Type of baseline function

TIME_ITV_START: int = 0                   # Start time interval for simulation
TIME_HORIZON: int = 100                   # Time horizon for simulation

PROCESS_NUM: int = 100_000                # Number of processes to simulate

END_T: int = 200                          # End time for estimation
NUM_SEQ: int = 100                        # Number of sequences for estimation

# Discretisation parameters (discretisation.py)

DISCRETISE_STEP: float = 1.0              # Discretisation step = Delta








       