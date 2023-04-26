# -*- coding: utf-8 -*-

""" Hawkes variables module

File containing all hawkes variables.

"""

from main import args

# Hawkes process hyper-parameters generation parameters (hyperparameters.py)

MIN_ITV_BETA: float = 1.0 if not args.min_itv_beta else args.min_itv_beta                # Beta minimum interval
MAX_ITV_BETA: float = 3.0 if not args.max_itv_beta else args.max_itv_beta                # Beta maximum interval

MIN_ITV_ETA: float = 0.05 if not args.min_itv_eta else args.min_itv_eta                  # Eta minimum interval
MAX_ITV_ETA: float = 0.8  if not args.max_itv_eta else args.max_itv_eta                  # Eta maximum interval

EXPECTED_ACTIVITY: int = 500 if not args.expected_activity else args.expected_activity   # Total number of expected events 
STD: float = 50 if not args.std else args.std                                            # Standard deviation for generating epsilon

# Hawkes Process simulation/estimation parameters (hawkes.py, discretisation.py)

KERNEL: str = 'exp' if not args.kernel else args.kernel                                  # Type of kernel function
BASELINE: str = 'const' if not args.baseline else args.baseline                          # Type of baseline function

TIME_ITV_START: int = 0 if not args.time_itv_start else args.time_itv_start              # Start time interval for simulation
TIME_HORIZON: int = 100 if not args.time_horizon else args.time_horizon                  # Time horizon for simulation

PROCESS_NUM: int = 160_000 if not args.process_num else args.process_num                 # Number of processes to simulate

END_T: int = 200 if not args.end_t else args.end_t                                       # End time for estimation
NUM_SEQ: int = 100 if not args.num_seq else args.num_seq                                 # Number of sequences for estimation

# Discretisation parameters (discretisation.py)

DISCRETISE_STEP: float = 1.0 if not args.discretise_step else args.discretise_step       # Discretise step = Delta








       