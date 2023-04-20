#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Utils test module

File containing all utils test functions

"""

import os
from typing import List, TypedDict
from unittest.mock import patch

import numpy as np
import pandas as pd

from UTILS.utils import write_csv, read_csv, write_parquet, read_parquet, timer, profiling
import VARIABLES.preprocessing_var as prep
import VARIABLES.evaluation_var as eval


# CSV file writing test function

def test_write_csv(data: List[dict] = [{'name': 'John', 'age': 28}, {'name': 'Jane', 'age': 32}], test_file: str = 'write_test.csv') -> None:

    """
    Test function for CSV file writing
    
    Args:
        data (List[dict], optional): List of dictionaries to be written to file (default: [{'name': 'John', 'age': 28}, {'name': 'Jane', 'age': 32}])
        test_file (str, optional): Filename for test file (default: 'write_test.csv')
        
    Returns:
        None: This function does not return anything
        
    Raises:
        AssertionError: Unexpected results
    """
    
    # Initialized expected results
    expected_content = "name,age\nJohn,28\nJane,32\n"
    
    # Called function
    write_csv(data=data, filename=test_file, mode='w', encoding='utf-8', filepath=prep.FILEPATH)
    
    # Red test file
    with open(os.path.join(prep.FILEPATH, test_file), 'r') as file:
        content = file.read()
    
    # Asserted results
    assert content == expected_content


# CSV file reading test function

def test_read_csv(filename: str = 'read_test.csv') -> None:
    
    """
    Test function for CSV file reading
    
    Args:
        filename (str, optional): Filename for test file (default: 'read_test.csv')
        
    Returns:
        None: This function does not return anything
        
    Raises:
        AssertionError: Unexpected results
    """

    # Initialized CSV file
    with open(filename, 'w') as f:
        f.write('col1,col2,col3\n')
        f.write('1,2,3\n')
        f.write('4,5,6\n')

    # Expected DataFrame
    expected_df = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32), columns=['col1', 'col2', 'col3'])

    # Called function
    actual_df = read_csv(filename)

    # Checked dataframe
    pd.testing.assert_frame_equal(actual_df, expected_df)

    # Cleaned up test file
    os.remove(filename)


# Parquet file writing test function

@patch("UTILS.utils.write_parquet")
def test_write_parquet(mock_write, data: TypedDict = {'a': [1, 2, 3], 'b': [4, 5, 6]}, filename: str = 'write_test.parquet', columns: List = ['a', 'b'], compression: str = 'SNAPPY') -> None:

    """
    Test function for Parquet file wrtiting
    
    Args:
        data (TypedDict, optional): Test data contents (default: {'a': [1, 2, 3], 'b': [4, 5, 6]})
        filename (str, optional): Filename for test file (default: 'write_test.parquet')
        columns (List, optional): Test data columns (default: ['a', 'b'])
        compression (str, optional): Compression type (default: 'SNAPPY')
        
    Returns:
        None: This function does not return anything
        
    Raises:
        AssertionError: Unexpected results
    """
        
    # Called function
    write_parquet(data, filename, columns, compression)

    # Asserted mock
    mock_write.assert_called_once_with(os.path.join(prep.FILEPATH, filename), 
                                       pd.DataFrame(data, columns=columns, dtype=np.float32), compression=compression)


# Parquet file reading test function

@patch("UTILS.utils.read_parquet")
def test_read_parquet(mock_read, data: pd.DataFrame = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']}), filename: str = 'read_test.parquet') -> None:
    
    """
    Test function for Parquet file reading
    
    Args:
        mock_read (MagicMock): Mock for read_parquet function
        data (TypedDict, optional): Test data contents (default: pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']}))
        filename (str, optional): Filename for test file (default: 'read_test.parquet')
        
    Returns:
        None: This function does not return anything
        
    Raises:
        AssertionError: Unexpected results
    """
        
    # Initialized mock
    mock_read.return_value.to_pandas.return_value = data

    # Called function
    result = read_parquet(filename)

    # Asserted type/result
    assert isinstance(result, pd.DataFrame)
    assert result.equals(data)


# Timer decorator test function

@patch('builtins.perf_counter', return_value=0.5)
@patch('builtins.process_time', return_value=0.1)
def test_timer(mock_perf_counter, mock_process_time, capsys):

    """
    Test function for timer decorator
    
    Args:
        mock_perf_counter (MagicMock): Mock for perf_counter function
        mock_process_time (MagicMock): Mock for process_time function
        capsys (Object): Capture IO results
        
    Returns:
        None: This function does not return anything
        
    Raises:
        AssertionError: Unexpected results
    """

    # Called decorator
    @timer
    def test_func(n: int):
        return sum(range(n))

    # Tested it
    test_func(10000)

    # Asserted print output
    assert capsys.readouterr().out == "Execution time (test_func): 0.000000s - CPU time: 0.000000s (Repetition: 7 - Iteration: 10)\n"


# Pytorch Tensorboard Profiling test function

def test_profiling(enable: bool = True):

    """
    Test function for Pytorch Tensorboard Profiling
    
    Args:
        enable (bool, optional): Flag indicating whether profiling is enabled or not (default: True)
        
    Returns:
        None: This function does not return anything
        
    Raises:
        AssertionError: Unexpected results
    """

    # Initialized parameters
    def my_func(x):
        return x + 1

    # Defined expected output
    expected_output = 4

    # Initialized mock
    @patch("torch.utils.tensorboard.SummaryWriter")
    def test_mock_summary_writer(mock_summary_writer):

        # Called decorator and function
        decorated_func = profiling(func=my_func, enable=enable)
        output = decorated_func(3)

        # Asserted mock
        mock_summary_writer.assert_called_once_with(eval.LOGDIPROF)

        # Asserted results
        assert output == expected_output