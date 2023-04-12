#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Dataset module

File containing DL preprocessing function.

"""

import torch
import numpy as np
from typing import Tuple
from torch.utils.data import DataLoader, TensorDataset

import VARIABLES.variables as var


# Input data
X = np.random.randn(var.INPUT_SIZE, 1).astype(np.float32)
Y = np.random.randn(var.INPUT_SIZE, var.OUTPUT_SIZE).astype(np.float32)


# Splitting function

def split_data(X: np.ndarray, Y: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    # Initialized sizing
    val_size = int(len(X) * var.VAL_RATIO)
    test_size = int(len(X) * var.TEST_RATIO)
    train_size = len(X) - val_size - test_size

    # Train/Val/Test split
    train_X, val_X, test_X = torch.tensor(X[:train_size], dtype=torch.float32), torch.tensor(X[train_size:train_size+val_size], dtype=torch.float32), torch.tensor(X[train_size+val_size:], dtype=torch.float32)
    train_Y, val_Y, test_Y = torch.tensor(Y[:train_size], dtype=torch.float32), torch.tensor(Y[train_size:train_size+val_size], dtype=torch.float32), torch.tensor(Y[train_size+val_size:], dtype=torch.float32)

    # Moved tensors to CPU/GPU
    train_X, val_X, test_X = train_X.to(var.DEVICE), val_X.to(var.DEVICE), test_X.to(var.DEVICE)
    train_Y, val_Y, test_Y = train_Y.to(var.DEVICE), val_Y.to(var.DEVICE), test_Y.to(var.DEVICE)

    return train_X, train_Y, val_X, val_Y, test_X, test_Y


# Dataset creation function

def create_datasets(train_X: torch.Tensor, train_Y: torch.Tensor, val_X: torch.Tensor, val_Y: torch.Tensor, test_X: torch.Tensor, test_Y: torch.Tensor) -> Tuple[TensorDataset, TensorDataset, TensorDataset]:

    # Datasets creation 
    train_dataset = TensorDataset(train_X, train_Y)
    val_dataset = TensorDataset(val_X, val_Y)
    test_dataset = TensorDataset(test_X, test_Y)

    return train_dataset, val_dataset, test_dataset


# Data Loaders creation function

def create_data_loaders(train_dataset: TensorDataset, val_dataset: TensorDataset, test_dataset: TensorDataset) -> Tuple[DataLoader, DataLoader, DataLoader]:
    
    # Data Loaders creation (speed up loading process with drop_last, num_workers, pin_memory)
    train_loader = DataLoader(train_dataset, batch_size=var.BATCH_SIZE, shuffle=var.SHUFFLE, drop_last=var.DROP_LAST, num_workers=var.NUM_WORKERS, pin_memory=var.PIN_MEMORY)
    val_loader = DataLoader(val_dataset, batch_size=var.BATCH_SIZE, shuffle=var.SHUFFLE, drop_last=var.DROP_LAST, num_workers=var.NUM_WORKERS, pin_memory=var.PIN_MEMORY) 
    test_loader = DataLoader(test_dataset, batch_size=var.BATCH_SIZE, shuffle=var.SHUFFLE, drop_last=var.DROP_LAST, num_workers=var.NUM_WORKERS, pin_memory=var.PIN_MEMORY)

    return train_loader, val_loader, test_loader

