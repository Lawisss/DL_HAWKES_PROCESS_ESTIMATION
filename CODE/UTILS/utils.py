# -*- coding: utf-8 -*-

"""Utils module

File containing all utils functions used in other modules (python files)

"""

import os 
from functools import wraps
from typing import List, Callable

import pandas as pd
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.profiler import schedule, tensorboard_trace_handler
from torch.profiler import profile, ProfilerActivity

import VARIABLES.preprocessing_var as prep
import VARIABLES.evaluation_var as eval


# CSV file writing function

def write_csv(data: List[dict], filename: str = '', mode: str = 'w', encoding: str = 'utf-8') -> None:

    """
    Written dictionaries list to a CSV file

    Args:
        data (List[dict]): Dictionaries list, where each dictionary represents row in CSV file
        filename (str): Filename to write data to. If not specified, empty string is used
        mode (str): Mode to open file in. Defaults to 'w' (write mode)
        encoding (str): Encoding to use when writing to file. Defaults to 'utf-8'

    Returns:
        None: Function does not return anything

    Raises:
        IOError: If there is error writing to the file
    """

    try:
        if not isinstance(data, list):
            data = [data]

        # Written and field names initialisation
        with open(f"{os.path.join(prep.FILEPATH, filename)}", mode=mode, encoding=encoding) as file:
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

    """
    Red CSV file and loaded as a DataFrame

    Args:
        filename (str): Filename to read
        delimiter (str, optional): Delimiter used to separate fields in the file. Defaults to ','
        mode (str, optional): Mode in which file is opened. Defaults to 'r'
        encoding (str, optional): Character encoding used to read file. Defaults to 'utf-8'

    Returns:
        pd.DataFrame: DataFrame containing file contents
    """

    try:
        with open(filepath=f"{os.path.join(prep.FILEPATH, filename)}", mode=mode, encoding=encoding) as file:

            # Extracted headers
            headers = next(file).strip().split(delimiter)

            # Extracted rows
            rows = np.array(list(map(lambda line: line.strip().split(delimiter), file)), dtype=np.float32)
                
        return pd.DataFrame(rows, columns=headers, dtype=np.float32)
    
    except IOError as e:
        print(f"Cannot read the file: {e}.")


# Pytorch Tensorboard Profiling 

def profiling(func: Callable = None, enable: bool = False) -> Callable:

    """
    Decorator function for profiling models using TensorBoard

    Args:
        func (Callable): Function to be decorated
        enable (bool): Flag indicating whether profiling is enabled or not. Defaults to False.

    Returns:
        Callable: Decorated function
    """
    
    # Executed only when decorating function
    def prof_decorator(func: Callable) -> Callable:

        # Decorated function information conservation
        @wraps(func)

        # Wrapper called when decorated function called and return decorated function result 
        def wrapper(*args, **kwargs):

            # Initialized Tensorboard
            writer = SummaryWriter(f"{os.path.join(eval.LOGDIPROF, eval.RUN_NAME)}")

            # Activated profiling
            if enable:
                
                # Defined profiler options
                profiler_options = {"activities": [ProfilerActivity.CPU, ProfilerActivity.CUDA],
                                    "schedule": schedule(wait=eval.WAIT, warmup=eval.WARMUP, active=eval.ACTIVE, repeat=eval.REPEAT, skip_first=eval.SKIP_FIRST),
                                    "on_trace_ready": tensorboard_trace_handler(eval.LOGDIPROF),
                                    "record_shapes": eval.RECORD_SHAPES,
                                    "profile_memory": eval.PROFILE_MEMORY,
                                    "with_stack": eval.WITH_STACK,
                                    "with_flops": eval.WITH_FLOPS,
                                    "with_modules": eval.WITH_MODULES,
                                    "use_cuda": prep.DEVICE}

                # Started profiling
                with profile(**profiler_options) as prof:

                    result = func(*args, **kwargs)

                    # Added profiling traces/results (average events + group by operator name/input shapes/stack) 
                    for trace in prof.key_averages(group_by_input_shape=eval.GROUP_BY_INPUT_SHAPE, 
                                                   group_by_stack_n=eval.GROUP_BY_STACK_N).table(sort_by=eval.SORT_BY,
                                                                                                 row_limit=eval.ROW_LIMIT):
                        writer.add_scalar(trace.key, trace.value, 0)

                    # Ended profiling and exported stack traces/collected traces
                    if prof.with_stack:
                        prof.export_stacks(f"{os.path.join(eval.LOGDIPROF, eval.RUN_NAME)}.txt", 
                                           "self_cuda_time_total" if prep.DEVICE == "cuda" else "self_cpu_time_total")

                    prof.export_chrome_trace(f"{os.path.join(eval.LOGDIPROF, eval.RUN_NAME)}.json")
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