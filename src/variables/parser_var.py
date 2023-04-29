# -*- coding: utf-8 -*-

"""Parser variables module

File containing all parser arguments.

"""

from typing import List

import variables.hawkes_var as hwk
import variables.mlp_var as mlp
import variables.eval_var as eval
import variables.prep_var as prep


# Parser parameters (parser_args.py)

PROG: str  = "Main"                                                             # Program name for command-line usage
USAGE: str  = "%(prog)s [options]"                                              # Message to display when invoking --help option
DESCRIPTION: str  = "Arguments Parser"                                          # Brief description of the program's functionality

ARG_GROUPS: List = [                                                            # List of groups containing command-line arguments
        
    {'name': 'hawkes_params', 
     'description': 'Hawkes process hyper-parameters generation parameters',
     'args': [
        {'name': '--min_itv_beta', 'type': float, 'nargs': 1, 'default': hwk.MIN_ITV_BETA, 'help': 'Beta minimum interval'},
        {'name': '--max_itv_beta', 'type': float, 'nargs': 1, 'default': hwk.MAX_ITV_BETA, 'help': 'Beta maximum interval'},
        {'name': '--min_itv_eta', 'type': float, 'nargs': 1, 'default': hwk.MIN_ITV_ETA, 'help': 'Eta minimum interval'},
        {'name': '--max_itv_eta', 'type': float, 'nargs': 1, 'default': hwk.MAX_ITV_ETA, 'help': 'Eta maximum interval'},
        {'name': '--expected_activity', 'type': int, 'nargs': 1, 'default': hwk.EXPECTED_ACTIVITY, 'help': 'Total number of expected events'},
        {'name': '--std', 'type': float, 'nargs': 1, 'default': hwk.STD, 'help': 'Standard deviation for generating epsilon'}]},

    {'name': 'hawkes_simulation_params', 
     'description': 'Hawkes Process simulation/estimation parameters',
     'args': [
        {'name': '--kernel', 'type': str, 'nargs': 1, 'default': hwk.KERNEL, 'help': 'Type of kernel function'},
        {'name': '--baseline', 'type': str, 'nargs': 1, 'default': hwk.BASELINE, 'help': 'Type of baseline function'},
        {'name': '--time_itv_start', 'type': int, 'nargs': 1, 'default': hwk.TIME_ITV_START, 'help': 'Start time interval for simulation'},
        {'name': '--time_horizon', 'type': int, 'nargs': 1, 'default': hwk.TIME_HORIZON, 'help': 'Time horizon for simulation'},
        {'name': '--process_num', 'type': int, 'nargs': 1, 'default': hwk.PROCESS_NUM, 'help': 'Number of processes to simulate'},
        {'name': '--end_t', 'type': int, 'nargs': 1, 'default': hwk.END_T, 'help': 'End time for estimation'},
        {'name': '--num_seq', 'type': int, 'nargs': 1, 'default': hwk.NUM_SEQ, 'help': 'Number of sequences for estimation'}]},

    {'name': 'discretisation_params', 
     'description': 'Discretisation parameters',
     'args': [
        {'name': '--discretise_step', 'type': float, 'nargs': 1, 'default': hwk.DISCRETISE_STEP, 'help': 'Discretise step = Delta'}]},

    {'name': 'data_params', 
     'description': 'Dataset parameters', 
     'args': [
        {'name': '--val_ratio', 'type': float, 'nargs': 1, 'default': prep.VAL_RATIO, 'help': 'Validation data ratio'},
        {'name': '--test_ratio', 'type': float, 'nargs': 1, 'default': prep.TEST_RATIO, 'help': 'Test data ratio'}]},

    {'name': 'loader_params', 
     'description': 'Data loader parameters',
     'args': [
        {'name': '--batch_size', 'type': int, 'nargs': 1, 'default': prep.BATCH_SIZE, 'help': 'Training batch size'},
        {'name': '--shuffle', 'type': bool, 'nargs': 1, 'default': prep.SHUFFLE, 'help': 'Shuffle training data'},
        {'name': '--drop_last', 'type': bool, 'nargs': 1, 'default': prep.DROP_LAST, 'help': 'Drop last batch if smaller than batch_size'},
        {'name': '--num_workers', 'type': int, 'nargs': 1, 'default': prep.NUM_WORKERS, 'help': 'Number of workers for data loading'},
        {'name': '--pin_memory', 'type': bool, 'nargs': 1, 'default': prep.PIN_MEMORY, 'help': 'Pin memory to accelerate GPU transfer'}]},

    {'name': 'device_params', 
     'description': 'Device parameters', 
     'args': [{'name': '--device', 'type': str, 'nargs': 1, 'default': str(prep.DEVICE), 'help': 'Device used for computation'}]},

    {'name': 'file_params', 
     'description': 'Writing/Loading parameters', 
     'args': [
         {'name': '--dirpath', 'type': str, 'nargs': 1, 'default': prep.DIRPATH, 'help': 'Path directory where results are saved'},
         {'name': '--default_dir', 'type': str, 'nargs': 1, 'default': prep.DEFAULT_DIR, 'help': "Default parquet file folder"},
         {'name': '--logdirun', 'type': str, 'nargs': 1, 'default': eval.LOGDIRUN, 'help': "Tensorboard logs directory for each run"},
         {'name': '--train_dir', 'type': str, 'nargs': 1, 'default': eval.TRAIN_DIR, 'help': "Training parquet file folder"},
         {'name': '--test_dir', 'type': str, 'nargs': 1, 'default': eval.TEST_DIR, 'help': "Testing parquet file folder"},
         {'name': '--best_model_dir', 'type': str, 'nargs': 1, 'default': eval.BEST_MODEL_DIR, 'help': "Best model folder"},
         {'name': '--run_name', 'type': str, 'nargs': 1, 'default': eval.RUN_NAME, 'help': "Name for current run based on timestamp/hostname"},
         {'name': '--logdiprof', 'type': str, 'nargs': 1, 'default': eval.LOGDIPROF, 'help': "Profiling results directory"}]},

    {'name': 'mlp_params', 
     'description': 'MLP parameters', 
     'args': [
        {'name': '--input_size', 'type': int, 'nargs': 1, 'default': mlp.INPUT_SIZE, 'help': 'MLP input size'},
        {'name': '--hidden_size', 'type': int, 'nargs': 1, 'default': mlp.HIDDEN_SIZE, 'help': 'Number of neurons in hidden layers'},
        {'name': '--output_size', 'type': int, 'nargs': 1, 'default': mlp.OUTPUT_SIZE, 'help': 'MLP output size'},
        {'name': '--num_hidden_layers', 'type': int, 'nargs': 1, 'default': mlp.NUM_HIDDEN_LAYERS, 'help': 'MLP number of hidden layers'},
        {'name': '--l2_reg', 'type': float, 'nargs': 1, 'default': mlp.L2_REG, 'help': 'L2 regularization parameter'},
        {'name': '--learning_rate', 'type': float, 'nargs': 1, 'default': mlp.LEARNING_RATE, 'help': 'Learning rate for optimizer'},
        {'name': '--max_epochs', 'type': int, 'nargs': 1, 'default': mlp.MAX_EPOCHS, 'help': 'Maximum number of epochs for training'},
        {'name': '--filename_best_model', 'type': str, 'nargs': 1, 'default': mlp.FILENAME_BEST_MODEL, 'help': 'Filename to save best model'},
        {'name': '--early_stop_patience', 'type': int, 'nargs': 1, 'default': mlp.EARLY_STOP_PATIENCE, 'help': 'Epochs without improvement before early stopping'},
        {'name': '--early_stop_delta', 'type': float, 'nargs': 1, 'default': mlp.EARLY_STOP_DELTA, 'help': 'Minimum reduction of val_loss to consider improvement'}]},

    {'name': 'summary_params', 
     'description': 'MLP Summary parameters', 
     'args': [
        {'name': '--summary_model', 'type': str, 'nargs': 1, 'default': mlp.SUMMARY_MODEL, 'help': 'Model name used for summary'},
        {'name': '--summary_mode', 'type': str, 'nargs': 1, 'default': mlp.SUMMARY_MODE, 'help': 'Summary modes: train/eval'},
        {'name': '--summary_verbose', 'type': int, 'nargs': 1, 'default': mlp.SUMMARY_VERBOSE, 'help': 'Level of verbosity in summary'},
        {'name': '--summary_col_names', 'type': str, 'nargs':'+', 'default': mlp.SUMMARY_COL_NAMES, 'help': 'Summary columns names'}]},

    {'name': 'profile_params', 
     'description': 'Profiling parameters', 
     'args': [
        {'name': '--activities', 'type': list, 'nargs': '+', 'default': eval.ACTIVITIES, 'help': "List of profiling activities to perform"},
        {'name': '--wait', 'type': int, 'nargs': 1, 'default': eval.WAIT, 'help': "Time (in seconds) to wait before starting profiling"},
        {'name': '--warmup', 'type': int, 'nargs': 1, 'default': eval.WARMUP, 'help': "Time (in seconds) for warming up before profiling"},
        {'name': '--active', 'type': int, 'nargs': 1, 'default': eval.ACTIVE, 'help': "Time (in seconds) for profiling"},
        {'name': '--repeat', 'type': int, 'nargs': 1, 'default': eval.REPEAT, 'help': "Number of times to repeat profiling"},
        {'name': '--skip_first', 'type': int, 'nargs': 1, 'default': eval.SKIP_FIRST, 'help': "Number of first profiling results to discard"},
        {'name': '--record_shapes', 'type': bool, 'nargs': 1, 'default': eval.RECORD_SHAPES, 'help': "Record tensor shapes in profiling output"},
        {'name': '--profile_memory', 'type': bool, 'nargs': 1, 'default': eval.PROFILE_MEMORY, 'help': "Include memory profiling"},
        {'name': '--with_stack', 'type': bool, 'nargs': 1, 'default': eval.WITH_STACK, 'help': "Include function call stack in profiling output"},
        {'name': '--with_flops', 'type': bool, 'nargs': 1, 'default': eval.WITH_FLOPS, 'help': "Include FLOPS computation in profiling output"},
        {'name': '--with_modules', 'type': bool, 'nargs': 1, 'default': eval.WITH_MODULES, 'help': "Include profiling of module operations"},
        {'name': '--group_by_input_shape', 'type': bool, 'nargs': 1, 'default': eval.GROUP_BY_INPUT_SHAPE, 'help': "Group profiling output by tensor input shapes"},
        {'name': '--group_by_stack_n', 'type': int, 'nargs': 1, 'default': eval.GROUP_BY_STACK_N, 'help': "Stack frames number to include in function call stack"},
        {'name': '--sort_by', 'type': str, 'nargs': 1, 'default': eval.SORT_BY, 'help': "Sort profiling output by specified metric"},
        {'name': '--row_limit', "type": int, 'nargs': 1, 'default': eval.ROW_LIMIT, 'help': "Maximum number of rows to display in profiling output"}]}
]
