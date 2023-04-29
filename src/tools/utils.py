# -*- coding: utf-8 -*-

"""Utils module

File containing all utils functions used in other modules (python files)

"""

import os 
from functools import wraps, lru_cache
from time import perf_counter, process_time
from typing import List, Callable, TypedDict, Optional

import pandas as pd
import numpy as np
import fastparquet as fp
from torch.utils.tensorboard import SummaryWriter
from torch.profiler import schedule, tensorboard_trace_handler
from torch.profiler import profile, ProfilerActivity

import variables.eval_var as eval
import variables.prep_var as prep


# csv file writing function

def write_csv(data: List[dict], filename: str = '', mode: str = 'w', encoding: str = 'utf-8') -> None:

    """
    Written dictionaries list to a csv file

    Args:
        data (List[dict]): Dictionaries list (dictionary = csv row)
        filename (str, optional): csv filename
        mode (str, optional): Mode to open file in (default: 'w' (write mode))
        encoding (str, optional): Encoding to use when writing to file (default: 'utf-8')

    Returns:
        None: Function does not return anything

    Raises:
        IOError: Writing file error
    """

    try:
        if not isinstance(data, list):
            data = [data]

        # Written and field names initialisation
        with open(os.path.join(prep.DIRPATH, prep.DEFAULT_DIR, filename), mode=mode, encoding=encoding) as file:
            file.write(','.join(data[0].keys()))
            file.write('\n')
        
            # Lines iteration
            for row in data:
                file.write(','.join(str(x) for x in row.values()))
                file.write('\n')
        
        # Closed file    
        file.close()
                    
    except IOError as e:
        print(f"Cannot write csv file: {e}.")


# csv file reading function

def read_csv(filename: str, delimiter: str = ',', mode: str = 'r', encoding: str = 'utf-8') -> pd.DataFrame:

    """
    Red csv file and loaded as DataFrame

    Args:
        filename (str): Filename to read
        delimiter (str, optional): Delimiter used to separate fields in the file (default: ',')
        mode (str, optional): Mode in which file is opened (default: 'r')
        encoding (str, optional): Character encoding used to read file (default: 'utf-8')

    Returns:
        pd.DataFrame: File contents dataFrame

    Raises:
        IOError: Reading file error
    """

    try:
        with open(os.path.join(prep.DIRPATH, prep.DEFAULT_DIR, filename), mode=mode, encoding=encoding) as file:

            # Extracted headers
            headers = next(file).strip().split(delimiter)

            # Extracted rows
            rows = np.array(list(map(lambda line: line.strip().split(delimiter), file)), dtype=np.float32)
                
        return pd.DataFrame(rows, columns=headers, dtype=np.float32)
    
    except IOError as e:
        print(f"Cannot read csv file: {e}.")


# Parquet file writing function

def write_parquet(data: TypedDict, filename: str = '', columns: Optional[str] = None, compression: Optional[str] = None) -> None:

    """
    Written dictionary to parquet file

    Args:
        data (TypedDict): Saved dictionary
        filename (str, optional): Parquet filename (default: '')
        folder (str, optional): Sub-folder name in results folder (default: 'simulations')
        columns (bool, optional): Index column writing (default: None)
        compression (str, optional): Column compression type (default: None)

    Returns:
        None: Function does not return anything

    Raises:
        IOError: Writing file error
    """

    try:
        # Write parquet file from dataframe (index/compression checked)
        fp.write(os.path.join(prep.DIRPATH, prep.DEFAULT_DIR, filename), pd.DataFrame(data, columns=columns, dtype=np.float32), compression=compression)
        
    except IOError as e:
        print(f"Cannot write parquet file: {e}")


# Parquet file reading function

def read_parquet(filename: str) -> pd.DataFrame:

    """
    Red Parquet file and loaded as DataFrame

    Args:
        filename (str): Parquet filename

    Returns:
        Pandas dataframe: File contents dataFrame

    Raises:
        IOError: Reading file error
    """

    try:
        # Load Parquet file using fastparquet
        pf = fp.ParquetFile(os.path.join(prep.DIRPATH, prep.DEFAULT_DIR, filename))
        # Converted it in dataframe
        return pf.to_pandas(columns=pf.columns)

    except IOError as e:
        print(f"Cannot read parquet file: {e}.")


# Parquet to csv function

def parquet_to_csv(parquet_file: str = "test.parquet", csv_file: str = "test.csv", index: bool = False) -> None:

    """
    Parquet to CSV conversion function

    Args:
        parquet_file (str, optional): Parquet filename (default: "test.parquet")
        csv_file (str, optional): csv filename (default: "test.csv")
        index (bool, optional): Write row names (default: False)

    Returns:
        Pandas dataframe: File contents dataFrame

    """

    # Red parquet file
    df = pd.read_parquet(os.path.join(prep.DIRPATH, prep.DEFAULT_DIR, parquet_file))
    # Writtent csv file
    df.to_csv(os.path.join(prep.DIRPATH, prep.DEFAULT_DIR, csv_file), index=index)


# Time measurement function

def timer(func: Callable = None, n_iter: int = 10, repeats: int = 7, returned: bool = False) -> Callable:

    """
    Decorator function for time measurement

    Args:
        func (Callable): Function to be decorated
        n_iter (int, optional): Iterations to perform (default: 10)
        repeats (int, optional): Times to repeat iterations. (default: 7)
        returned (bool, optional): Flag indicating whether return function results or not (default: False)

    Returns:
        Callable: Decorated function
    """

    # Executed only when decorating function
    def timer_decorator(func: Callable) -> Callable:

        # Stored references to run functions
        perf_timer = perf_counter
        process_timer = process_time
        func_name = func.__name__

        # Decorated function information conservation
        @wraps(func)

        # Wrapper called when decorated function called and return decorated function result 
        @lru_cache(maxsize=None)
        def wrapper(*args, **kwargs):
            
            # Time initialization
            total_time = 0
            total_process_time = 0

            for _ in range(repeats):

                # Started time
                start_process_time = process_timer()
                start_time = perf_timer()

                for _ in range(n_iter):
                    # Call decorated function
                    result = func(*args, **kwargs)

                # Ended time
                end_time = perf_timer()
                end_process_time = process_timer()
            
                # Total performance/process time computation
                total_time += round(end_time - start_time, 6)
                total_process_time += round(end_process_time - start_process_time, 6)

            # Total performance/process time
            avg_time = total_time / repeats
            avg_process_time = total_process_time / repeats

            print(f"Execution time ({func_name}): {avg_time:.6f}s - CPU time: {avg_process_time:.6f}s (Repetition: {repeats} - Iteration: {n_iter})")

            # Returned fonction results
            if returned:
                return result
        
        return wrapper
    
    # Decorator call, ex: @timer / Factory call, ex: @timer()
    return timer_decorator(func) if func else timer_decorator


# Pytorch Tensorboard Profiling 

def profiling(func: Callable = None, enable: bool = False) -> Callable:

    """
    Decorator function for profiling models using TensorBoard

    Args:
        func (Callable, optional): Function to be decorated
        enable (bool, optional): Flag indicating whether profiling is enabled or not (default: False)

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
            writer = SummaryWriter(os.path.join(eval.LOGDIPROF, eval.RUN_NAME))

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
    
    # Decorator call, ex: @profiling / Factory call, ex: @profiling()
    return prof_decorator(func) if func else prof_decorator