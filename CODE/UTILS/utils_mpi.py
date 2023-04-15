# -*- coding: utf-8 -*-

"""Utils MPI module

File containing all MPI utils functions used in other modules (python files)

"""

import os 
from typing import List

import numpy as np
import pandas as pd
from mpi4py import MPI

import VARIABLES.preprocessing_var as prep

# Parallelized CSV file writing function

def write_csv(data: List[dict], filename: str = '', mode: str = 'w', encoding: str = 'utf-8') -> None:

    """
    Written dictionaries list to CSV file in parallel using MPI

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
        with open(filepath=f"{os.path.join(prep.FILEPATH, filename)}", mode=mode, encoding=encoding) as file:
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
        print(f"Cannot read the file: {e}.")


# Parallelized CSV file reading function

def read_csv(filename: str, delimiter: str = ',', mode: str = 'r', encoding: str = 'utf-8') -> pd.DataFrame:

    """
    Red CSV file in parallel using MPI

    Args:
        filename (str): CSV filename to read
        delimiter (str, optional): Delimiter used in CSV file. Defaults to ','
        mode (str, optional): Mode used to open CSV file. Defaults to 'r'
        encoding (str, optional): Encoding used to read CSV file. Defaults to 'utf-8'

    Returns:
           pandas.DataFrame: DataFrame containing data from CSV file
    """

    # Initialized MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    try:
        with open(filepath=f"{os.path.join(prep.FILEPATH, filename)}", mode=mode, encoding=encoding) as file:

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
        print(f"Cannot read the file: {e}.")
