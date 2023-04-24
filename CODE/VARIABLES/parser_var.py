# -*- coding: utf-8 -*-

"""Parser variables module

File containing all parser arguments.

"""

from typing import List

import VARIABLES.preprocessing_var as prep
import VARIABLES.mlp_var as mlp


# Parser parameters (parser_args.py)

PROG: str  = "MAIN"                                                             # Program name for command-line usage
USAGE: str  = "%(prog)s [options]"                                              # Message to display when invoking --help option
DESCRIPTION: str  = "Training model"                                            # Brief description of the program's functionality

ARG_GROUPS: List = [                                                            # List of groups containing command-line arguments
        
    {'name': 'data_params', 
     'description': 'Dataset parameters', 
     'args': [
        {'name': '--val_ratio', 'type': float, 'default': prep.VAL_RATIO, 'help': 'Validation data ratio'},
        {'name': '--test_ratio', 'type': float, 'default': prep.TEST_RATIO, 'help': 'Test data ratio'}]},

    {'name': 'loader_params', 
     'description': 'Data loader parameters',
     'args': [
        {'name': '--batch_size', 'type': int, 'default': prep.BATCH_SIZE, 'help': 'Training batch size'},
        {'name': '--shuffle', 'type': bool, 'default': prep.SHUFFLE, 'help': 'Shuffle training data'},
        {'name': '--drop_last', 'type': bool, 'default': prep.DROP_LAST, 'help': 'Drop last batch if smaller than batch_size'},
        {'name': '--num_workers', 'type': int, 'default': prep.NUM_WORKERS, 'help': 'Number of workers for data loading'},
        {'name': '--pin_memory', 'type': bool, 'default': prep.PIN_MEMORY, 'help': 'Pin memory to accelerate GPU transfer'}]},

    {'name': 'device_params', 
     'description': 'Device parameters', 
     'args': [{'name': '--device', 'type': str, 'default': str(prep.DEVICE), 'help': 'Device used for computation'}]},

    {'name': 'file_params', 
     'description': 'Writing/Loading parameters', 
     'args': [{'name': '--filepath', 'type': str, 'default': prep.FILEPATH, 'help': 'Path directory where results are saved'}]},

    {'name': 'mlp_params', 
     'description': 'MLP parameters', 
     'args': [
        {'name': '--input_size', 'type': int, 'default': mlp.INPUT_SIZE, 'help': 'MLP input size'},
        {'name': '--hidden_size', 'type': int, 'default': mlp.HIDDEN_SIZE, 'help': 'Number of neurons in hidden layers'},
        {'name': '--output_size', 'type': int, 'default': mlp.OUTPUT_SIZE, 'help': 'MLP output size'},
        {'name': '--num_hidden_layers', 'type': int, 'default': mlp.NUM_HIDDEN_LAYERS, 'help': 'MLP number of hidden layers'},
        {'name': '--l2_reg', 'type': float, 'default': mlp.L2_REG, 'help': 'L2 regularization parameter'},
        {'name': '--learning_rate', 'type': float, 'default': mlp.LEARNING_RATE, 'help': 'Learning rate for optimizer'},
        {'name': '--max_epochs', 'type': int, 'default': mlp.MAX_EPOCHS, 'help': 'Maximum number of epochs for training'},
        {'name': '--filename_best_model', 'type': str, 'default': mlp.FILENAME_BEST_MODEL, 'help': 'Filename to save best model'},
        {'name': '--early_stop_patience', 'type': int, 'default': mlp.EARLY_STOP_PATIENCE, 'help': 'Epochs without improvement before early stopping'},
        {'name': '--early_stop_delta', 'type': float, 'default': mlp.EARLY_STOP_DELTA, 'help': 'Minimum reduction of val_loss to consider improvement'}]},

    {'name': 'summary_params', 
     'description': 'MLP Summary parameters', 
     'args': [
        {'name': '--summary_model', 'type': str, 'default': mlp.SUMMARY_MODEL, 'help': 'Model name used for summary'},
        {'name': '--summary_mode', 'type': str, 'default': mlp.SUMMARY_MODE, 'help': 'Summary modes: train/eval'},
        {'name': '--summary_verbose', 'type': int, 'default': mlp.SUMMARY_VERBOSE, 'help': 'Level of verbosity in summary'},
        {'name': '--summary_col_names', 'type': str, 'nargs':'+', 'default': mlp.SUMMARY_COL_NAMES, 'help': 'Summary columns names'}]}
]