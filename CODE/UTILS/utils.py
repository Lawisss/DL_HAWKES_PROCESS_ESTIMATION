# -*- coding: utf-8 -*-

"""Utils module

File containing all utils functions used in other python files (modules).

"""

# CSV file writing function

def write_csv(data, filepath='', mode='w', encoding='utf-8'):
    
    try:
        if not isinstance(data, list):
            data = [data]

        # Write and field names initialisation
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