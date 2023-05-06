# -*- coding: utf-8 -*-

"""Utils module

File containing all utils functions used in other modules (python files)

"""

import os 
from functools import wraps, lru_cache
from time import perf_counter, process_time
from typing import List, Callable, TypedDict, Optional

import polars as pl
from torch.utils.tensorboard import SummaryWriter
from torch.profiler import schedule, tensorboard_trace_handler
from torch.profiler import profile

import variables.eval_var as eval
import variables.prep_var as prep


# csv file writing function

def write_csv(data: List[dict], filename: str = '', mode: str = 'w', encoding: str = 'utf-8', folder: str = "simulations", args: Optional[Callable] = None) -> None:

    """
    Written dictionaries list to a csv file

    Args:
        data (List[dict]): Dictionaries list (dictionary = csv row)
        filename (str, optional): csv filename (default: '')
        mode (str, optional): Mode to open file in (default: 'w' (write mode))
        encoding (str, optional): Encoding to use when writing to file (default: 'utf-8')
        folder (str, optional): Sub-folder name in results folder (default: "simulations")
        args (Callable, optional): Arguments if you use run.py instead of tutorial.ipynb (default: None)

    Returns:
        None: Function does not return anything

    Raises:
        IOError: Writing file error
    """

    # Default parameters
    default_params = {"dirpath": prep.DIRPATH}

    # Initialized parameters
    dict_args = {k: getattr(args, k, v) for k, v in default_params.items()}

    try:
        if not isinstance(data, list):
            data = [data]

        # Written and field names initialisation
        with open(os.path.join(dict_args['dirpath'], folder, filename), mode=mode, encoding=encoding) as file:
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

def read_csv(filename: str, separator: str = ',', folder: str = "simulations", args: Optional[Callable] = None) -> pl.DataFrame:

    """
    Red csv file and loaded as DataFrame

    Args:
        filename (str): Filename to read
        separator (str, optional): Delimiter used to separate fields in the file (default: ',')
        folder (str, optional): Sub-folder name in results folder (default: 'simulations')
        args (Callable, optional): Arguments if you use run.py instead of tutorial.ipynb (default: None)

    Returns:
        pl.DataFrame: File contents dataFrame

    Raises:
        IOError: Reading file error
    """

    # Default parameters
    default_params = {"dirpath": prep.DIRPATH}

    # Initialized parameters
    dict_args = {k: getattr(args, k, v) for k, v in default_params.items()}

    try:
        # Polars read_csv function
        return pl.read_csv(os.path.join(dict_args['dirpath'], folder, filename), separator=separator)
    
    except IOError as e:
        print(f"Cannot read csv file: {e}.")


# Parquet file writing function

def write_parquet(data: TypedDict, filename: str = '', folder: str = "simulations", schema: Optional[str] = None, compression: Optional[str] = None, args: Optional[Callable] = None) -> None:

    """
    Written dictionary to parquet file

    Args:
        data (TypedDict): Saved dictionary
        filename (str, optional): Parquet filename (default: '')
        folder (str, optional): Sub-folder name in results folder (default: 'simulations')
        schema (bool, optional): Index column writing (default: None)
        compression (str, optional): Column compression type (default: None)
        args (Callable, optional): Arguments if you use run.py instead of tutorial.ipynb (default: None)

    Returns:
        None: Function does not return anything

    Raises:
        IOError: Writing file error
    """

    # Default parameters
    default_params = {"dirpath": prep.DIRPATH}

    # Initialized parameters
    dict_args = {k: getattr(args, k, v) for k, v in default_params.items()}

    try:
        # Write parquet file from dataframe (index/compression checked)
        pl.DataFrame(data, schema).write_parquet(os.path.join(dict_args['dirpath'], folder, filename), compression=compression)

    except IOError as e:
        print(f"Cannot write parquet file: {e}")


# Parquet file reading function

def read_parquet(filename: str, folder: str = "simulations", args: Optional[Callable] = None) -> pl.DataFrame:

    """
    Red Parquet file and loaded as DataFrame

    Args:
        filename (str): Parquet filename
        folder (str, optional): Sub-folder name in results folder (default: 'simulations')
        args (Callable, optional): Arguments if you use run.py instead of tutorial.ipynb (default: None)

    Returns:
        Polars dataframe: File contents dataFrame

    Raises:
        IOError: Reading file error
    """

    # Default parameters
    default_params = {"dirpath": prep.DIRPATH}

    # Initialized parameters
    dict_args = {k: getattr(args, k, v) for k, v in default_params.items()}

    try:
        # Polars read_parquet function
        return pl.read_parquet(os.path.join(dict_args['dirpath'], folder, filename))

    except IOError as e:
        print(f"Cannot read parquet file: {e}.")


# Parquet to csv function

def parquet_to_csv(parquet_file: str = "test.parquet", csv_file: str = "test.csv", folder: str = "simulations", args: Optional[Callable] = None) -> None:

    """
    Parquet to CSV conversion function

    Args:
        parquet_file (str, optional): Parquet filename (default: "test.parquet")
        csv_file (str, optional): csv filename (default: "test.csv")
        folder (str, optional): Sub-folder name in results folder (default: 'simulations')
        args (Callable, optional): Arguments if you use run.py instead of tutorial.ipynb (default: None)

    Returns:
        None: Function does not return anything

    """

    # Default parameters
    default_params = {"dirpath": prep.DIRPATH}

    # Initialized parameters
    dict_args = {k: getattr(args, k, v) for k, v in default_params.items()}

    # Red parquet file
    df = pl.read_parquet(os.path.join(dict_args['dirpath'], folder, parquet_file))
    # Written csv file
    df.write_csv(os.path.join(dict_args['dirpath'], folder, csv_file))


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

def profiling(func: Callable = None, enable: bool = False, args: Optional[Callable] = None) -> Callable:

    """
    Decorator function for profiling models using TensorBoard

    Args:
        func (Callable, optional): Function to be decorated
        enable (bool, optional): Flag indicating whether profiling is enabled or not (default: False)
        args (Callable, optional): Arguments if you use run.py instead of tutorial.ipynb (default: None)

    Returns:
        Callable: Decorated function
    """
    
    # Executed only when decorating function
    def prof_decorator(func: Callable) -> Callable:

        # Decorated function information conservation
        @wraps(func)

        # Wrapper called when decorated function called and return decorated function result 
        def wrapper(*_args, **kwargs):
            
            # Default parameters
            default_params = {"logdiprof": eval.LOGDIPROF, 
                              "run_name": eval.RUN_NAME,
                              "activities": eval.ACTIVITIES, 
                              "wait": eval.WAIT,
                              "warmup": eval.WARMUP, 
                              "active": eval.ACTIVE,
                              "repeat": eval.REPEAT, 
                              "skip_first": eval.SKIP_FIRST,
                              "record_shapes": eval.RECORD_SHAPES, 
                              "profile_memory": eval.PROFILE_MEMORY,
                              "with_stack": eval.WITH_STACK, 
                              "with_flops": eval.WITH_FLOPS,
                              "with_modules": eval.WITH_MODULES, 
                              "use_cuda": prep.DEVICE,
                              "group_by_input_shape": eval.GROUP_BY_INPUT_SHAPE,
                              "group_by_stack_n": eval.GROUP_BY_STACK_N,
                              "sort_by": eval.SORT_BY,
                              "row_limit": eval.ROW_LIMIT}

            # Initialized parameters
            dict_args = {k: getattr(args, k, v) for k, v in default_params.items()}

            # Initialized Tensorboard
            writer = SummaryWriter(os.path.join(dict_args["logdiprof"], dict_args["run_name"]))

            # Activated profiling
            if enable:
                
                # Defined profiler options
                profiler_options = {"activities": dict_args["activities"],
                                    "schedule": schedule(wait=dict_args["wait"], warmup=dict_args["warmup"], active=dict_args["active"], repeat=dict_args["repeat"], skip_first=dict_args["skip_first"]),
                                    "on_trace_ready": tensorboard_trace_handler(dict_args["logdiprof"]),
                                    "record_shapes": dict_args["record_shapes"],
                                    "profile_memory": dict_args["profile_memory"],
                                    "with_stack": dict_args["with_stack"],
                                    "with_flops": dict_args["with_flops"],
                                    "with_modules": dict_args["with_modules"],
                                    "use_cuda": dict_args["use_cuda"]}

                # Started profiling
                with profile(**profiler_options) as prof:

                    result = func(*_args, **kwargs)

                    # Added profiling traces/results (average events + group by operator name/input shapes/stack) 
                    for trace in prof.key_averages(group_by_input_shape=dict_args["group_by_input_shape"], 
                                                   group_by_stack_n=dict_args["group_by_stack_n"]).table(sort_by=dict_args["sort_by"], row_limit=dict_args["row_limit"]):
                        writer.add_scalar(trace.key, trace.value, 0)

                    # Ended profiling and exported stack traces/collected traces
                    if prof.with_stack:
                        prof.export_stacks(f"{os.path.join(dict_args['logdiprof'], dict_args['run_name'])}.txt", 
                                           "self_cuda_time_total" if prep.DEVICE == "cuda" else "self_cpu_time_total")

                    prof.export_chrome_trace(f"{os.path.join(dict_args['logdiprof'], dict_args['run_name'])}.json")
                    writer.add_scalar("Training Time (Total)", prof.total_average().cpu().numpy(), 0)

                    # Closed SummaryWriter
                    writer.close()
            else:
                # No profiling
                result = func(*_args, **kwargs)
                
            return result

        return wrapper
    
    # Decorator call, ex: @profiling / Factory call, ex: @profiling()
    return prof_decorator(func) if func else prof_decorator