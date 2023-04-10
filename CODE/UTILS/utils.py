# -*- coding: utf-8 -*-

"""Utils module

File containing all utils functions used in other modules (python files).

"""


import pandas as pd
import numpy as np
from typing import List

import VARIABLES.variables as var

# CSV file writing function

def write_csv(data: List[dict], filename: str='', mode: str='w', encoding: str='utf-8') -> None:
    
    try:
        if not isinstance(data, list):
            data = [data]

        # Written and field names initialisation
        with open(f"{var.FILEPATH}{filename}", mode=mode, encoding=encoding) as file:
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

def read_csv(filename: str, delimiter: str=',', mode: str='r', encoding: str='utf-8') -> pd.DataFrame:

    try:
        with open(f"{var.FILEPATH}{filename}", mode=mode, encoding=encoding) as file:

            # Extracted headers
            headers = next(file).strip().split(delimiter)

            # Extracted rows
            rows = np.array(list(map(lambda line: line.strip().split(delimiter), file)), dtype=np.float64)
                
        return pd.DataFrame(rows, columns=headers, dtype=np.float64)
    
    except IOError as e:
        print(f"Cannot read the file: {e}.")