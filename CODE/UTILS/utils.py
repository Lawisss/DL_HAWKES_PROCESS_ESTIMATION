# -*- coding: utf-8 -*-

"""Utils module

File containing all utils functions used in other modules (python files).

"""

import os 

import torch
import pandas as pd
import numpy as np
from typing import List
from functools import wraps
from torch.utils.tensorboard import SummaryWriter
from torch.profiler import schedule, tensorboard_trace_handler
from torch.profiler import profile, ProfilerActivity

import VARIABLES.variables as var


# CSV file writing function

def write_csv(data: List[dict], filename: str = '', mode: str = 'w', encoding: str = 'utf-8') -> None:
    
    try:
        if not isinstance(data, list):
            data = [data]

        # Written and field names initialisation
        with open(filepath=f"{os.path.join(var.FILEPATH, filename)}", mode=mode, encoding=encoding) as file:
            file.write(','.join(data[0].keys()))
            file.write('\n')
        
            # Lines iteration
            for row in data:
                file.write(','.join(str(x) for x in row.values()))
                file.write('\n')
        
        # Closed file    
        file.close()
                    
    except IOError as e:
        print(f"Cannot read the file: {e}.")


# CSV file reading function

def read_csv(filename: str, delimiter: str = ',', mode: str = 'r', encoding: str = 'utf-8') -> pd.DataFrame:

    try:
        with open(filepath=f"{os.path.join(var.FILEPATH, filename)}", mode=mode, encoding=encoding) as file:

            # Extracted headers
            headers = next(file).strip().split(delimiter)

            # Extracted rows
            rows = np.array(list(map(lambda line: line.strip().split(delimiter), file)), dtype=np.float32)
                
        return pd.DataFrame(rows, columns=headers, dtype=np.float32)
    
    except IOError as e:
        print(f"Cannot read the file: {e}.")


# Pytorch Tensorboard Profiling 

def profiling(func=None, enable=False):
    
    # Executed only when decorating function
    def prof_decorator(func):

        # Decorated function information conservation
        @wraps(func)

        # Wrapper called when decorated function called and return decorated function result 
        def wrapper(*args, **kwargs):

            # Initialized Tensorboard
            writer = SummaryWriter(f"{os.path.join(var.LOGDIPROF, var.RUN_NAME)}")

            # Activated profiling
            if enable:
                
                # Defined profiler options
                profiler_options = {"activities": [ProfilerActivity.CPU, ProfilerActivity.CUDA],
                                    "schedule": schedule(wait=var.WAIT, warmup=var.WARMUP, active=var.ACTIVE, repeat=var.REPEAT, skip_first=var.SKIP_FIRST),
                                    "on_trace_ready": tensorboard_trace_handler(var.LOGDIPROF),
                                    "record_shapes": var.RECORD_SHAPES,
                                    "profile_memory": var.PROFILE_MEMORY,
                                    "with_stack": var.WITH_STACK,
                                    "with_flops": var.WITH_FLOPS,
                                    "with_modules": var.WITH_MODULES,
                                    "use_cuda": var.DEVICE}

                # Started profiling
                with profile(**profiler_options) as prof:

                    result = func(*args, **kwargs)

                    # Added profiling traces/results (average events + group by operator name/input shapes/stack) 
                    for trace in prof.key_averages(group_by_input_shape=var.GROUP_BY_INPUT_SHAPE, 
                                                   group_by_stack_n=var.GROUP_BY_STACK_N).table(sort_by=var.SORT_BY,
                                                                                                row_limit=var.ROW_LIMIT):
                        writer.add_scalar(trace.key, trace.value, 0)

                    # Ended profiling and exported stack traces/collected traces
                    if prof.with_stack:
                        prof.export_stacks(f"{os.path.join(var.LOGDIPROF, var.RUN_NAME)}.txt", 
                                           "self_cuda_time_total" if var.DEVICE == "cuda" else "self_cpu_time_total")

                    prof.export_chrome_trace(f"{os.path.join(var.LOGDIPROF, var.RUN_NAME)}.json")
                    writer.add_scalar("Training Time (Total)", prof.total_average().cpu().numpy(), 0)

                    # Closed SummaryWriter
                    writer.close()
            else:
                # No profiling
                result = func(*args, **kwargs)
                
            return result

        return wrapper
    
    # Decorator creator (profiling) return decorator
    if func:
        # Actual decorator call, ex: @cached_property
        return prof_decorator(func)
    else:
        # Factory call, ex: @cached_property()
        return prof_decorator