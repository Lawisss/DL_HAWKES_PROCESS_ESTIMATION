#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""DL test module

File containing DL model test function

"""

from typing import Callable, List
from unittest.mock import patch, Mock

import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils import parameters_to_vector

from DL.mlp import MLP, MLPTrainer
import VARIABLES.mlp_var as mlp
import VARIABLES.preprocessing_var as prep


# Forward test function

@patch('VARIABLES.mlp_var.INPUT_SIZE', 10)
@patch('VARIABLES.mlp_var.HIDDEN_SIZE', 20)
@patch('VARIABLES.mlp_var.NUM_HIDDEN_LAYERS', 2)
@patch('VARIABLES.mlp_var.OUTPUT_SIZE', 1)
def test_forward(input_data: torch.Tensor = torch.rand((32, mlp.INPUT_SIZE))) -> None:

    """
    Test function for MLP forward function 

    Args:
        input_data (torch.Tensor, optional): Input data (default: torch.rand((32, mlp.INPUT_SIZE)))

    Returns:
        None: Function does not return anything

    Raises:
        AssertionError: Unexpected results
    """

    # Called function
    output = MLP().forward(input_data)

    # Asserted shape
    assert output.shape == (32, mlp.OUTPUT_SIZE)


# Summary test function

@patch('DL.mlp.summary')
def test_summary_model(mock_summary, trainer: Callable = MLPTrainer()) -> None:

    """
    Test function for MLP summary

    Args:
        mock_summary (MagicMock): Mock for summary function
        trainer (Callable, optional): MLP Trainer (default: MLPTrainer())

    Returns:
        None: Function does not return anything

    Raises:
        AssertionError: Unexpected results
    """

    # Called function
    summary_str = trainer.summary_model()

    # Asserted mock
    mock_summary.assert_called_once_with(trainer.model,
                                         input_size=mlp.INPUT_SIZE,
                                         input_data=[prep.BATCH_SIZE, mlp.INPUT_SIZE],
                                         batch_dim=prep.BATCH_SIZE,
                                         col_names=mlp.SUMMARY_COL_NAMES,
                                         device=prep.DEVICE,
                                         mode=mlp.SUMMARY_MODE,
                                         verbose=mlp.SUMMARY_VERBOSE)

    # Asserted results
    assert summary_str == f"{trainer.SUMMARY_MODEL:^30} Summary"


# Automatic gradients backpropagation test function

@patch.object(torch.nn.functional, 'cross_entropy')
def test_run_epoch(mock_loss, data: List = [(torch.randn(3, 4), torch.tensor([1])) for _ in range(10)]) -> None:

    """
    Test function for one epoch running

    Args:
        mock_loss (MagicMock): Mock for loss function
        data (List, optional): Test data (default: [(torch.randn(3, 4), torch.tensor([1])) for _ in range(10)])

    Returns:
        None: Function does not return anything

    Raises:
        AssertionError: Unexpected results
    """

    # Loaded data
    loader = DataLoader(data, batch_size=2)

    # Initialized parameters
    model = torch.nn.Linear(4, 2)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    train_loss = 0

    # Called function
    mlp = MLP(model, criterion, optimizer, train_loss)
    avg_loss = mlp.run_epoch(loader)

    # Asserted types/values
    assert isinstance(avg_loss, float)
    assert avg_loss >= 0

    # Asserted parameters
    assert model.weight.grad is not None and model.bias.grad is not None
    assert torch.norm(parameters_to_vector(model.parameters()), p=2) > 0

    # Asserted mock
    assert mock_loss.call_count == 5


# Evaluation test function

@patch('DL.mlp.MLP.forward')
def test_evaluate(mock_forward, model: Callable = MLP(), val_dataset: TensorDataset = TensorDataset(torch.randn(4, 3, 32, 32), torch.randint(0, 10, size=(4,)))) -> None:

    """
    Test function for model evaluation

    Args:
        mock_forward (MagicMock): Mock for forward function
        model (Callable, optional): Model to evaluate (default: MLP())
        val_dataset (List, optional): Validation dataset (default: TensorDataset = TensorDataset(torch.randn(4, 3, 32, 32), torch.randint(0, 10, size=(4,)))))

    Returns:
        None: Function does not return anything

    Raises:
        AssertionError: Unexpected results
    """

    # Set mock return
    mock_forward.return_value = torch.randn(2, 10)

    # Called function
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    val_loss = model.evaluate(val_loader)

    # Asserted type
    assert isinstance(val_loss, float)


# Early stopping test function

def test_early_stopping() -> None:

    """
    Test function for early stopping

    Args:
        None: Function does not contain parameters

    Returns:
        None: Function does not return anything

    Raises:
        AssertionError: Unexpected results
    """

    # Initialized parameters
    MLPTrainer().val_loss = 1.0
    best_loss = 2.0
    no_improve_count = 5

    # Asserted early stopping
    assert MLPTrainer().early_stopping(best_loss, no_improve_count) == True


# Loading model test function

@patch('torch.load')
def test_load_model(mock_load) -> None:

    """
    Test function for model prediction

    Args:
        mock_load (Mock): Mock for prediction function

    Returns:
        None: Function does not return anything

    Raises:
        AssertionError: Unexpected results
    """

    # Initialized parameters
    expected_filepath = "my_file_path"
    expected_message = "Best model loading (my_best_model.pth)..."
    mock_load.return_value = "my_model_state_dict"
    
    # Called function
    actual_message = MLPTrainer().load_model()
    
    # Asserted mock
    mock_load.assert_called_once_with(expected_filepath)

    # Asserted results
    assert actual_message == expected_message


# Prediction test function

def test_predict(model: Callable = Mock()) -> None:

    """
    Test function for model prediction

    Args:
        mock_forward (Mock): Mock for prediction function

    Returns:
        None: Function does not return anything

    Raises:
        AssertionError: Unexpected results
    """

    # Initialized parameters
    expected_val_y_pred = torch.Tensor([[0.5, 0.3], [0.4, 0.2]])
    expected_val_eta = 0.45
    expected_val_mu = 0.25
    model.predict.return_value = (expected_val_y_pred, expected_val_eta, expected_val_mu)
    
    # Called function
    val_x = torch.Tensor([[1.0, 2.0], [3.0, 4.0]])
    val_y_pred, val_eta, val_mu = model.predict(val_x)
    
    # Asserted types/values
    assert val_y_pred == expected_val_y_pred and val_eta == expected_val_eta and val_mu == expected_val_mu
    assert isinstance(val_y_pred, torch.Tensor) and isinstance(val_eta, float) and isinstance(val_mu, float)


# Training test function

def test_train_model(x_train = torch.randn((100, 10)), y_train = torch.randn((100, 1)), x_val = torch.randn((20, 10)), y_val = torch.randn((20, 1))) -> None:

    """
    Test function for model evaluation

    Args:
        x_train (MagicMock): Training features (default: torch.randn((100, 10)))
        y_train (Callable, optional): Training labels (default: torch.randn((100, 1)))
        x_val (List, optional): Validation features (default: torch.randn((20, 10)))
        y_val (Callable, optional): Validation labels (default: torch.randn((20, 1)))

    Returns:
        None: Function does not return anything

    Raises:
        AssertionError: Unexpected results
    """

    # Initialized parameters
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=10)
    val_dataset = TensorDataset(x_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=10)

    # Called function
    train_losses, val_losses = MLPTrainer().train_model(train_loader, val_loader, x_val)

    # Asserted types
    assert isinstance(MLPTrainer().model, torch.nn.Module) and isinstance(train_losses, np.ndarray) and isinstance(val_losses, np.ndarray) and isinstance(x_val, torch.Tensor)