#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Dataset test module

File containing DL preprocessing test function

"""

import unittest
from unittest.mock import patch, MagicMock

import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from PREPROCESSING.dataset import split_data, create_datasets, create_data_loaders

# Dataset test class

class DatasetTests(unittest.TestCase):

    # Splitting test function

    @patch("VARIABLES.preprocessing_var.VAL_RATIO", 0.2)
    @patch("VARIABLES.preprocessing_var.TEST_RATIO", 0.1)
    @patch("VARIABLES.preprocessing_var.DEVICE", "cpu")
    def test_split_data(self):

        # Defined parameters
        x = np.random.rand(100, 5)
        y = np.random.rand(100, 1)

        # Called function under test
        train_x, train_y, val_x, val_y, test_x, test_y = split_data(x, y)

        # Checked outputs types/shapes
        self.assertTupleEqual((train_x.shape, train_y.shape, val_x.shape, val_y.shape, test_x.shape, test_y.shape), 
                              ((60, 5), (60, 1), (20, 5), (20, 1), (20, 5), (20, 1)))
    

    # Dataset creation test function

    def test_create_datasets(self):

        # Create some fake data
        train_x = torch.randn(100, 10)
        train_y = torch.randn(100, 1)
        val_x = torch.randn(50, 10)
        val_y = torch.randn(50, 1)
        test_x = torch.randn(25, 10)
        test_y = torch.randn(25, 1)

        # Mock the TensorDataset constructor
        tensor_dataset_mock = MagicMock()

        # Called function under test
        with patch('my_module.TensorDataset', tensor_dataset_mock):
            train_dataset, val_dataset, test_dataset = create_datasets(train_x, train_y, val_x, val_y, test_x, test_y)

        # Asserted arguments
        self.assertEqual(tensor_dataset_mock.call_count, 3)
        tensor_dataset_mock.assert_any_call(train_x, train_y)
        tensor_dataset_mock.assert_any_call(val_x, val_y)
        tensor_dataset_mock.assert_any_call(test_x, test_y)

        # Asserted function returns
        self.assertIsInstance(train_dataset, TensorDataset)
        self.assertIsInstance(val_dataset, TensorDataset)
        self.assertIsInstance(test_dataset, TensorDataset)


if __name__ == '__main__':
    unittest.main()