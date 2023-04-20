#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from unittest.mock import Mock, patch, call

import numpy as np
import Hawkes as hk
from pytest import approx

import VARIABLES.hawkes_var as hwk
from HAWKES.hyperparameters import hyper_params_simulation
from HAWKES.hawkes import hawkes_simulation, hawkes_simulations, hawkes_estimation
from HAWKES.discretisation import discretise, temp_func, find_stepsize, jump_times


# Hyperparameters simulation test function

@patch("UTILS.utils.write_parquet")
def test_hyper_params_simulation(mock_write_parquet) -> None:

    """
    Test function for hyperparameters simulation
       
    Args:
        mock_write_parquet (MagicMock): Mock for write_parquet function

    Returns:
        None: Function does not return anything
    """

    # Called mock function
    result, alpha, beta, mu = hyper_params_simulation(filename="hyperparams_test.parquet")

    # Asserted types/shapes/results
    assert all(isinstance(arr, np.ndarray) for arr in [result, alpha, beta, mu])
    np.testing.assert_equal(result.shape, alpha.shape, beta.shape, mu.shape, ((hwk.PROCESS_NUM, 3), (hwk.PROCESS_NUM,), (hwk.PROCESS_NUM,), (hwk.PROCESS_NUM,)))
    assert np.all(alpha == mu / np.exp(beta))

    # Asserted function calling
    mock_write_parquet.assert_called_once_with({"alpha": alpha, "beta": beta, "mu": mu}, filename="hyperparams_test.parquet")


# Hawkes simulation test function

def test_hawkes_simulation() -> None:

    """
    Test function for Hawkes simulation

    Args:
        None: This function contains no arguments
       
    Returns:
        None: This function does not return anything
    """

    # Initialized parameters
    expected_times = np.array([1.0, 2.0, 3.0])
    expected_params = {"mu": 0.1, "alpha": 0.5, "beta": 10.0}

    # Mock simulator
    mock_simulator = Mock()
    mock_simulator.simulate.return_value = expected_times

    # Called mock function 
    simulator, times = hawkes_simulation(params=expected_params)

    # Asserted types
    assert isinstance(simulator, hk.simulator)
    assert simulator.kernel == hwk.KERNEL
    assert simulator.baseline == hwk.BASELINE
    assert simulator.parameters == expected_params

    # Asserted simulated times
    assert np.array_equal(times, expected_times)


# Hawkes simulations test function

@patch('HAWKES.hawkes.hawkes_simulation')
def test_hawkes_simulations(mock_hawkes_simulation) -> None:

    """
    Test function for hawkes simulations function

    Args:
        mock_hawkes_simulation (MagicMock): Mock for hawkes_simulation function

    Returns:
        None: Function does not return anything
    """

    # Initialized parameters
    alpha = np.array([[0.2, 0.3], [0.1, 0.2]])
    beta = np.array([[1.0, 0.5], [0.5, 0.8]])
    mu = np.array([0.1, 0.2])

    # Defined outputs
    expected_output = np.zeros((2, 10))

    # Set return value for mock
    mock_hawkes_simulation.return_value = (None, np.zeros(10))

    # Called function
    actual_output = hawkes_simulations(alpha, beta, mu)

    # Asserted actual output
    assert np.array_equal(actual_output, expected_output)


# Hawkes estimation test function

@patch('HAWKES.hawkes.hawkes_estimation')
def test_hawkes_estimation(mock_hawkes_estimation) -> None:

    """
    Test function for hawkes estimation function

    Args:
        mock_hawkes_estimation (MagicMock): Mock for hawkes_estimation function

    Returns:
        None: Function does not return anything
    """

    # Mock simulator
    mock_hawkes_estimation.return_value = (np.array([0.1, 0.5, 1.2, 2.0]), 
                                           {'Event(s)': 4, 'Parameters': [0.1, 0.2], 'Branching Ratio': 0.8, 'Log-Likelihood': -10.0, 'AIC': 20.0}, 
                                           np.array([0.1, 0.5, 0.8, 1.0]), 
                                           np.array([0.4, 0.7, 0.2, 0.3]))

    # Called function
    t_pred, metrics, t_transform, interval_transform = hawkes_estimation(np.array([0.1, 0.5, 1.2, 2.0]))

     # Asserted results/shapes/types
    assert np.all(np.diff(t_pred) >= 0)
    assert np.all((t_transform >= 0) & (t_transform <= 1))

    assert len(t_transform) == len(interval_transform)
    assert isinstance(metrics, dict)
    assert all(key in metrics for key in ['Event(s)', 'Parameters', 'Branching Ratio', 'Log-Likelihood', 'AIC'])


# Discretisation test function

@patch('UTILS.utils.write_parquet')
def test_discretise(mock_write_parquet) -> None:

    """
    Test function for Hawkes processes discretisation function

    Args:
        mock_write_parquet (MagicMock): Mock for write_parquet function

    Returns:
        None: Function does not return anything
    """

    # Called function
    counts = discretise(np.random.rand(5, 1000))

    # Asserted type/shape
    assert counts.shape == (5, 1000)
    assert counts.dtype == np.float32

    # Asserted function calling
    mock_write_parquet.assert_called_once_with(counts, columns=np.arange(hwk.TIME_HORIZON, dtype=np.int32).astype(str), filename='binned_hawkes_simulations.parquet')


# Minimum stepsize test function

def test_temp_func() -> None:

    """
    Test function for minimum stepsize between events computation

    Args:
        None: This function contains no arguments

    Returns:
        None: Function does not return anything
    """

    # Initialized parameters/Asserted results (case n°1)
    jump_times = np.array([1.2, 2.3, 4.5, 5.6])
    expected_result = 0.9
    result = temp_func(jump_times)

    assert result == expected_result

    # Initialized parameters/Asserted results (case n°2)
    jump_times = np.array([])
    expected_result = 10.0
    result = temp_func(jump_times)

    assert result == expected_result


# temp_func(x, hwk.TIME_HORIZON) minimum test function

@patch('HAWKES.discretisation.temp_func')
def test_find_stepsize(mock_temp_func) -> None:

    """
    Test function for temp_func(x, hwk.TIME_HORIZON) minimum computation

    Args:
        None: This function contains no arguments

    Returns:
        None: Function does not return anything
    """

    # Initialized parameters
    jump_times = np.array([1, 2, 3, 4, 5])

    # Mock configuration
    mock_temp_func.side_effect = [1, 2, 3, 4]

    # Called function
    result = find_stepsize(jump_times)

    # Asserted calls
    mock_temp_func.assert_has_calls([call(1, hwk.TIME_HORIZON), call(2, hwk.TIME_HORIZON),
                                     call(3, hwk.TIME_HORIZON), call(4, hwk.TIME_HORIZON)])
    
    # Global minimum = 1
    assert result == 1  


# Jump times test function

@patch("numpy.random.uniform", return_value=np.array([0.25, 0.4, 0.8, 0.6]))
def test_jump_times(mock_uniform) -> None:

    """
    Test function for jump times computation

    Args:
        None: This function contains no arguments

    Returns:
        None: Function does not return anything
    """

    # Initialized parameters
    h = np.array([0, 1, 2, 0, 3, 0, 0, 1, 0])
    expected_result = np.array([0.25, 0.4, 0.4, 0.5, 0.6, 0.8, 0.8, 0.9])

    # Called function
    result = jump_times(h)

    # Asserted results
    assert result == approx(expected_result, rel=1e-6)

    # Asserted mock
    mock_uniform.assert_called_once_with(0.5, 0.6, size=(3,))