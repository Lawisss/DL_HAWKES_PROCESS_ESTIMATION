# -*- coding: utf-8 -*-

"""Utils module

File containing all utils functions used in other modules (python files).

"""


import pandas as pd
import numpy as np

# CSV file writing function

def write_csv(data, filepath='', mode='w', encoding='utf-8'):
    
    try:
        if not isinstance(data, list):
            data = [data]

        # Written and field names initialisation
        with open(filepath, mode=mode, encoding=encoding) as file:
            file.write(','.join(data[0].keys()))
            file.write('\n')
        
            # Lines iteration
            for row in data:
                file.write(','.join(str(x) for x in row.values()))
                file.write('\n')
        
        # Close file properly    
        file.close()
                    
    except IOError as e:
        print(f"Cannot read the file: {e}.")


# CSV file reading function

def read_csv(filepath, delimiter=',', mode='r', encoding='utf-8'):

    try:
        with open(filepath, mode=mode, encoding=encoding) as file:

            # Extract headers
            headers = next(file).strip().split(delimiter)

            # Extract rows using a list comprehension
            rows = [line.strip().split(delimiter) for line in file]
                
        return pd.DataFrame(rows, columns=headers, dtype=np.float64)
    
    except IOError as e:
        print(f"Cannot read the file: {e}.")