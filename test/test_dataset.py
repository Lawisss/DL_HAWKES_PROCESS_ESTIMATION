#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Dataset test module

File containing DL preprocessing test function

"""

import unittest

import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from preprocessing.dataset import split_data, create_datasets, create_data_loaders


# Test data splitting function

def test_split_data(x: np.ndarray = np.random.rand(100, 10), y: np.ndarray = np.random.rand(100, 1)) -> None:

    """
    Test function for split data function

    Args:
        x (np.ndarray, optional): Input features (default: np.random.rand(100, 10))
        y (np.ndarray, optional): Target values (default: np.random.rand(100, 10))

    Returns:
        None: Function does not return anything

    Raises:
        AssertionError: Unexpected results
    """

    # Called function
    train_x, train_y, val_x, val_y, test_x, test_y = split_data(x, y)

    # Asserted split sizes
    assert all(tensor.shape[0] == size for tensor, size in zip([train_x, val_x, test_x, train_y, val_y, test_y], 
                                                               [int(len(x) * 0.6), int(len(x) * 0.2), int(len(x) * 0.2),
                                                                int(len(x) * 0.6), int(len(x) * 0.2), int(len(x) * 0.2)]))

    # Asserted types
    assert all(isinstance(tensor, torch.Tensor) for tensor in (train_x, val_x, test_x, train_y, val_y, test_y))

    # Assserted dataset lengths
    assert len(train_x) + len(val_x) + len(test_x) == len(x)
    assert len(train_y) + len(val_y) + len(test_y) == len(y)

    # Asserted at least one element
    assert len(train_x) > 0 and len(val_x) > 0 and len(test_x) > 0

    # Asserted types
    assert all((train_x.dtype, val_x.dtype, test_x.dtype, train_y.dtype, val_y.dtype, test_y.dtype) == (torch.float32,) * 6)

    # Asserted device
    assert all(data.device in [torch.device("cpu"), torch.device("cuda")] for data in (train_x, val_x, test_x, train_y, val_y, test_y))

    # Asserted dimensions
    assert train_x.shape[1] == x.shape[1] and train_y.shape[1] == y.shape[1]


# Test dataset creation function

def test_create_datasets(train_x: torch.Tensor = torch.randn(100, 5), train_y: torch.Tensor = torch.randint(0, 2, (100,)), val_x: torch.Tensor = torch.randn(20, 5), val_y: torch.Tensor = torch.randint(0, 2, (20,)), test_x: torch.Tensor = torch.randn(30, 5), test_y: torch.Tensor = torch.randint(0, 2, (30,))) -> None:

    """
    Test function for dataset creation

    Args:
        train_x (torch.Tensor, optional): Input tensor for training data (default: torch.randn(100, 5))
        train_y (torch.Tensor, optional): Target tensor for training data (default: torch.randint(0, 2, (100,)))
        val_x (torch.Tensor, optional): Input tensor for validation data (default: torch.randn(20, 5))
        val_y (torch.Tensor, optional): Target tensor for validation data (default: torch.randint(0, 2, (20,)))
        test_x (torch.Tensor, optional): Input tensor for test data (default: torch.randn(30, 5))
        test_y (torch.Tensor, optional): Target tensor for test data (default: torch.randint(0, 2, (30,)))

    Returns:
        None: Function does not return anything

    Raises:
        AssertionError: Unexpected results
    """
    
    # Called function
    train_dataset, val_dataset, test_dataset = create_datasets(train_x, train_y, val_x, val_y, test_x, test_y)
    
    # Asserted types/lengths/shapes
    assert isinstance(train_dataset, TensorDataset) and isinstance(val_dataset, TensorDataset) and isinstance(test_dataset, TensorDataset)
    assert all(len(dataset) == len(x) for dataset, x in [(train_dataset, train_x), (val_dataset, val_x), (test_dataset, test_x)])
    assert all(torch.allclose(ds[i][0], x[i]) and ds[i][1] == y[i] for ds, x, y in [(train_dataset, train_x, train_y), (val_dataset, val_x, val_y), (test_dataset, test_x, test_y)] for i in range(len(ds)))


# Test dataset creation function

def test_create_data_loaders(x: torch.Tensor = torch.randn((100, 10)), y: torch.Tensor = torch.randint(0, 2, (100,))) -> None:

    """
    Test function for data loaders creation

    Args:
        x (torch.Tensor, optional): Input features (default: torch.randn((100, 10)))
        y (torch.Tensor, optional): Target values (default: torch.randint(0, 2, (100,)))

    Returns:
        None: Function does not return anything

    Raises:
        AssertionError: Unexpected results
    """

    # Initialized parameters
    dataset = TensorDataset(x, y)
    train_dataset, val_dataset, test_dataset = dataset, dataset, dataset

    # Called function
    train_loader, val_loader, test_loader = create_data_loaders(train_dataset, val_dataset, test_dataset)

    # Asserted types/lengths/values
    assert isinstance(train_loader, DataLoader) and isinstance(val_loader, DataLoader) and isinstance(test_loader, DataLoader)

    assert len(train_loader.dataset) == len(train_dataset)
    assert len(val_loader.dataset) == len(val_dataset)
    assert len(test_loader.dataset) == len(test_dataset)
    assert len(train_loader.dataset) == len(val_loader.dataset) == len(test_loader.dataset) == len(dataset)

    assert train_loader.batch_size == val_loader.batch_size == test_loader.batch_size == 32
    assert train_loader.shuffle == val_loader.shuffle == test_loader.shuffle == True
    assert train_loader.drop_last == val_loader.drop_last == test_loader.drop_last == True
    assert train_loader.num_workers == val_loader.num_workers == test_loader.num_workers == 4
    assert train_loader.pin_memory == val_loader.pin_memory == test_loader.pin_memory == True


if __name__ == '__main__':
    unittest.main()