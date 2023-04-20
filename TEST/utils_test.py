#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Utils test module

File containing all utils test functions

"""

import os
from typing import List
from unittest import mock

import numpy as np
import pandas as pd

from UTILS.utils import write_csv, read_csv
import VARIABLES.preprocessing_var as prep


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
