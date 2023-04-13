# -*- coding: utf-8 -*-

"""Evaluation variables module

File containing all evaluation variables.

"""

import os
import socket
from datetime import datetime

from typing import List
from torch.profiler import ProfilerActivity

# Tensorboard (Metrics evaluation - mlp.py)

LOGDIRUN: str = os.path.abspath("RESULTS/RUNS")                        # Tensorboard logs directory for each run
RUN_NAME: str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S") \
                + "_" + socket.gethostname().upper()                   # Name for current run based on timestamp/hostname

# Tensorboard (Fonction profiling - utils.py)

LOGDIPROF: str = os.path.abspath("RESULTS/PROFILING")                  # Profiling results directory 

ACTIVITIES: List[ProfilerActivity] = \
            [ProfilerActivity.CPU, ProfilerActivity.CUDA]              # CPU and CUDA profiling
WAIT: int = 1                                                          # Time (in seconds) to wait before starting profiling
WARMUP: int = 1                                                        # Time (in seconds) for warming up before profiling
ACTIVE: int = 2                                                        # Time (in seconds) for profiling
REPEAT: int = 0                                                        # Number of times to repeat profiling
SKIP_FIRST: int = 0                                                    # Number of first profiling results to discard
RECORD_SHAPES: bool = True                                             # Record tensor shapes in profiling output
PROFILE_MEMORY: bool = False                                           # Include memory profiling
WITH_STACK: bool = False                                               # Include function call stack in profiling output
WITH_FLOPS: bool = False                                               # Include FLOPS computation in profiling output
WITH_MODULES: bool = False                                             # Include profiling of module operations

GROUP_BY_INPUT_SHAPE: bool = False                                     # Group profiling output by tensor input shapes
GROUP_BY_STACK_N: int = 0                                              # Stack frames number to include in function call stack
SORT_BY: str = "cpu_time_total"                                        # Sort profiling output by specified metric
ROW_LIMIT: int = 10                                                    # Maximum number of rows to display in profiling output