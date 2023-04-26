# -*- coding: utf-8 -*-

"""Evaluation variables module

File containing all evaluation variables.

"""

import os
import socket
from typing import List
from datetime import datetime

from torch.profiler import ProfilerActivity

from main import args

# Tensorboard (Metrics evaluation - mlp.py)

LOGDIRUN: str = os.path.abspath("RESULTS/RUNS") if not args.logdirun else args.logdirun    # Tensorboard logs directory for each run

RUN_NAME: str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S") \
                + "_" + socket.gethostname().upper() \
                if not args.run_name else args.run_name                                    # Name for current run based on timestamp/hostname

# Tensorboard (Fonction profiling - utils.py)

LOGDIPROF: str = os.path.abspath("RESULTS/PROFILING") \
                 if not args.logdiprof else args.logdiprof                                 # Profiling results directory 

ACTIVITIES: List[ProfilerActivity] = \
            [ProfilerActivity.CPU, ProfilerActivity.CUDA] \
            if not args.activities else args.activities                                    # CPU and CUDA profiling

WAIT: int = 1 if not args.wait else args.wait                                              # Time (in seconds) to wait before starting profiling
WARMUP: int = 1 if not args.warmup else args.warmup                                        # Time (in seconds) for warming up before profiling
ACTIVE: int = 2 if not args.active else args.active                                        # Time (in seconds) for profiling
REPEAT: int = 0 if not args.repeat else args.repeat                                        # Number of times to repeat profiling
SKIP_FIRST: int = 0 if not args.skip_first else args.skip_first                            # Number of first profiling results to discard
RECORD_SHAPES: bool = True if not args.record_shapes else args.record_shapes               # Record tensor shapes in profiling output
PROFILE_MEMORY: bool = False if not args.profile_memory else args.profile_memory           # Include memory profiling
WITH_STACK: bool = False if not args.with_stack else args.with_stack                       # Include function call stack in profiling output
WITH_FLOPS: bool = False if not args.with_flops else args.with_flops                       # Include FLOPS computation in profiling output
WITH_MODULES: bool = False if not args.with_modules else args.with_modules                 # Include profiling of module operations

GROUP_BY_INPUT_SHAPE: bool = False if not args.group_by_input_shape \
                             else args.vgroup_by_input_shape                               # Group profiling output by tensor input shapes

GROUP_BY_STACK_N: int = 0 if not args.group_by_stack_n else args.group_by_stack_n          # Stack frames number to include in function call stack
SORT_BY: str = "cpu_time_total" if not args.sort_by else args.sort_by                      # Sort profiling output by specified metric
ROW_LIMIT: int = 10 if not args.row_limit else args.row_limit                              # Maximum number of rows to display in profiling output