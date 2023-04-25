#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Dataset module

File containing DL preprocessing function

"""

from typing import Tuple

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

import VARIABLES.preprocessing_var as prep

# Splitting function

def split_data(x: np.ndarray, y: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    """
    Splitted data into train, validation, and test sets

    Args:
        x (np.ndarray): Input features
        y (np.ndarray): target values

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            Training features, training targets, validation features, validation targets, test features, and test targets
    """

    # Converted data
    if isinstance(x, pd.DataFrame): x = torch.tensor(x.values, dtype=torch.float32).to(prep.DEVICE)
    if isinstance(y, pd.DataFrame): y = torch.tensor(y.values, dtype=torch.float32).to(prep.DEVICE)

    # Initialized sizing
    val_size = int(len(x) * prep.VAL_RATIO)
    test_size = int(len(x) * prep.TEST_RATIO)
    train_size = len(x) - val_size - test_size

    train_x, val_x, test_x = x[:train_size], x[train_size:train_size+val_size], x[train_size+val_size:]
    train_y, val_y, test_y = y[:train_size], y[train_size:train_size+val_size], y[train_size+val_size:]

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

def create_data_loaders(train_dataset: TensorDataset, val_dataset: TensorDataset, test_dataset: TensorDataset) -> Tuple[DataLoader, DataLoader, DataLoader]:
    
    """Created data loaders for training, validation, and testing sets

    Args:
        train_dataset (TensorDataset): Training dataset
        val_dataset (TensorDataset): Validation dataset
        test_dataset (TensorDataset): Testing dataset

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: Data loaders for the training, validation, and testing sets
    """
    
    # Data Loaders creation (speed up loading process with drop_last, num_workers, pin_memory)
    train_loader = DataLoader(train_dataset, batch_size=prep.BATCH_SIZE, shuffle=prep.SHUFFLE, drop_last=prep.DROP_LAST, num_workers=prep.NUM_WORKERS, pin_memory=prep.PIN_MEMORY)
    val_loader = DataLoader(val_dataset, batch_size=prep.BATCH_SIZE, shuffle=prep.SHUFFLE, drop_last=prep.DROP_LAST, num_workers=prep.NUM_WORKERS, pin_memory=prep.PIN_MEMORY) 
    test_loader = DataLoader(test_dataset, batch_size=prep.BATCH_SIZE, shuffle=prep.SHUFFLE, drop_last=prep.DROP_LAST, num_workers=prep.NUM_WORKERS, pin_memory=prep.PIN_MEMORY)

    return train_loader, val_loader, test_loader




