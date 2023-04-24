#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Dataset module

File containing DL preprocessing function

"""

from typing import Tuple

import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

import VARIABLES.preprocessing_var as prep
from UTILS.parser_args import argparser

# Splitting function

@argparser(parse_args=False, arg_groups=['data_params', 'device_params'])
def split_data(args_parsed, x: np.ndarray, y: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    """
    Splitted data into train, validation, and test sets

    Args:
        x (np.ndarray): Input features
        y (np.ndarray): target values

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            Training features, training targets, validation features, validation targets, test features, and test targets
    """

    val_ratio, test_ratio, device = (args_parsed.val_ratio, args_parsed.test_ratio, torch.device(args_parsed.device)) \
                                    if args_parsed else (prep.VAL_RATIO, prep.TEST_RATIO, prep.DEVICE)

    # Initialized sizing
    val_size = int(len(x) * val_ratio)
    test_size = int(len(x) * test_ratio)
    train_size = len(x) - val_size - test_size

    # Train/Val/Test split
    train_x, val_x, test_x = torch.tensor(x[:train_size], dtype=torch.float32), torch.tensor(x[train_size:train_size+val_size], dtype=torch.float32), torch.tensor(x[train_size+val_size:], dtype=torch.float32)
    train_y, val_y, test_y = torch.tensor(y[:train_size], dtype=torch.float32), torch.tensor(y[train_size:train_size+val_size], dtype=torch.float32), torch.tensor(y[train_size+val_size:], dtype=torch.float32)

    # Moved tensors to CPU/GPU
    train_x, val_x, test_x = train_x.to(device), val_x.to(device), test_x.to(device)
    train_y, val_y, test_y = train_y.to(device), val_y.to(device), test_y.to(device)

    return train_x, train_y, val_x, val_y, test_x, test_y


# Dataset creation function

def create_datasets(train_x: torch.Tensor, train_y: torch.Tensor, val_x: torch.Tensor, val_y: torch.Tensor, test_x: torch.Tensor, test_y: torch.Tensor) -> Tuple[TensorDataset, TensorDataset, TensorDataset]:

    """
    Created training, validation, and test sets from input tensors

    Args:
        train_x (torch.Tensor): Input tensor for training data
        train_y (torch.Tensor): Target tensor for training data
        val_x (torch.Tensor): Input tensor for validation data
        val_y (torch.Tensor): Target tensor for validation data
        test_x (torch.Tensor): Input tensor for test data
        test_y (torch.Tensor): Target tensor for test data

    Returns:
        Tuple[TensorDataset, TensorDataset, TensorDataset]: training, validation, and test sets
    """

    # Datasets creation 
    train_dataset = TensorDataset(train_x, train_y)
    val_dataset = TensorDataset(val_x, val_y)
    test_dataset = TensorDataset(test_x, test_y)

    return train_dataset, val_dataset, test_dataset


# Data Loaders creation function

@argparser(parse_args=False, arg_groups=['loader_params'])
def create_data_loaders(args_parsed, train_dataset: TensorDataset, val_dataset: TensorDataset, test_dataset: TensorDataset) -> Tuple[DataLoader, DataLoader, DataLoader]:
    
    """Created data loaders for training, validation, and testing sets

    Args:
        train_dataset (TensorDataset): Training dataset
        val_dataset (TensorDataset): Validation dataset
        test_dataset (TensorDataset): Testing dataset

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: Data loaders for the training, validation, and testing sets
    """
    
    batch_size, shuffle, drop_last, num_workers, pin_memory = (args_parsed.batch_size, args_parsed.shuffle, args_parsed.drop_last, args_parsed.num_workers, args_parsed.pin_memory) \
                                                              if args_parsed else (prep.BATCH_SIZE, prep.SHUFFLE, prep.DROP_LAST, prep.NUM_WORKERS, prep.PIN_MEMORY)
    
    # Data Loaders creation (speed up loading process with drop_last, num_workers, pin_memory)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers, pin_memory=pin_memory) 
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader




