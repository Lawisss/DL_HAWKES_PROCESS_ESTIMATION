# -*- coding: utf-8 -*-

""" Hawkes variables module

File containing all hawkes variables.

"""

# Hawkes process hyperparameters generation parameters (hyperparameters.py)

MIN_ITV_BETA: float = 1.0                 # Beta minimum interval (exponential kernel)
MAX_ITV_BETA: float = 3.0                 # Beta maximum interval (exponential kernel)

MIN_ITV_K: float = 0.05                   # K minimum interval (power law kernel)
MAX_ITV_K: float = 2.5                    # K maximum interval (power law kernel)

MIN_ITV_C: float = 1.0                    # C minimum interval (power law kernel)
MAX_ITV_C: float = 3.0                    # C maximum interval (power law kernel)

MIN_ITV_P: float = 1.0                    # P minimum interval
MAX_ITV_P: float = 0.5                    # P maximum interval

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








       