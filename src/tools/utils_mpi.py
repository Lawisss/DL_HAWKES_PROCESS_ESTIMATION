# -*- coding: utf-8 -*-

"""Utils MPI module

File containing all MPI utils functions used in other modules (python files)

"""

import os 
from typing import List, Optional, Callable

import numpy as np
import pandas as pd
from mpi4py import MPI

import variables.prep_var as prep

# Parallelized csv file writing function

def write_csv(data: List[dict], filename: str = '', mode: str = 'w', encoding: str = 'utf-8', folder: str = "simulations", args: Optional[Callable] = None) -> None:

    """
    Written dictionaries list to csv file in parallel using MPI

    Args:
        data (List[dict]): Dictionaries list, where each dictionary represents row in csv file
        filename (str, optional): Filename to write data to. If not specified, empty string is used
        mode (str, optional): Mode to open file in (default: 'w' (write mode))
        encoding (str, optional): Encoding to use when writing to file (default: 'utf-8')
        folder (str, optional): Sub-folder name in results folder (default: 'simulations')
        args (Callable, optional): Arguments if you use run.py instead of tutorial.ipynb

    Returns:
        None: Function does not return anything

    Raises:
        IOError: If there is error writing to the file
    """

    # Default parameters
    default_params = {"dirpath": prep.DIRPATH}

    # Initialized parameters
    dict_args = {k: getattr(args, k, v) for k, v in default_params.items()}

    # Initialized MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Splitted data into equal chunks based on processes number
    data_chunks = list(map(lambda i: data[i::size], range(size)))
    data_chunk = data_chunks[rank]

    try:
        if not isinstance(data_chunk, list):
            data_chunk = [data_chunk]

        # Written and field names initialisation (only rank 0 writes headers)
        with open(filepath=os.path.join(dict_args['dirpath'], folder, filename), mode=mode, encoding=encoding) as file:
            if rank == 0:
                file.write(','.join(data[0].keys()))
                file.write('\n')

            # Lines iteration
            for row in data_chunk:
                file.write(','.join(str(x) for x in row.values()))
                file.write('\n')

        # Synchronized all processes before closing
        comm.Barrier()

        # Only rank 0 closes file
        if rank == 0:
            file.close()

    except IOError as e:
        print(f"Cannot write csv file: {e}.")


# Parallelized csv file reading function

def read_csv(filename: str, delimiter: str = ',', mode: str = 'r', encoding: str = 'utf-8', folder: str = "simulations", args: Optional[Callable] = None) -> pd.DataFrame:

    """
    Red csv file in parallel using MPI

    Args:
        filename (str): csv filename to read
        delimiter (str, optional): Delimiter used in csv file (default: ',')
        mode (str, optional): Mode used to open csv file (default: 'r')
        encoding (str, optional): Encoding used to read csv file (default: 'utf-8')
        folder (str, optional): Sub-folder name in results folder (default: 'simulations')
        args (Callable, optional): Arguments if you use run.py instead of tutorial.ipynb

    Returns:
           pandas.DataFrame: DataFrame containing data from csv file
    """

    # Default parameters
    default_params = {"dirpath": prep.DIRPATH}

    # Initialized parameters
    dict_args = {k: getattr(args, k, v) for k, v in default_params.items()}

    # Initialized MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    try:
        with open(filepath=f"{os.path.join(dict_args['dirpath'], folder, filename)}", mode=mode, encoding=encoding) as file:

            # Determined processes sizes portion
            file_size = file.seek(0, 2)
            chunk_size = file_size // size
            remainder = file_size % size

            if rank == size - 1:
                chunk_size += remainder

            # Seeked chunk start for this process
            file.seek(rank * chunk_size)

            # Red chunk for this process
            chunk = file.read(chunk_size)

        # Splitted chunk into lines and extract data
        lines = chunk.split('\n')
        headers = lines[0].strip().split(delimiter)
        rows = np.array(list(map(lambda line: line.strip().split(delimiter), lines[1:])), dtype=np.float32)

        # Gathered and concatenated from all processes
        rows = comm.allgather(rows)
        rows = np.concatenate(rows)

        return pd.DataFrame(rows, columns=headers)

    except IOError as e:
        print(f"Cannot read csv file: {e}.")