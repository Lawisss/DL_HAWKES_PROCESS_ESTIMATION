#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import unittest
from unittest.mock import Mock, MagicMock, patch 

import numpy as np
import pandas as pd

import HAWKES.hawkes
import VARIABLES.hawkes_var as hwk
from UTILS.utils import read_parquet
from HAWKES.hyperparameters import hyper_params_simulation
from HAWKES.hawkes import hawkes_simulation, hawkes_simulations, hawkes_estimation

class HawkesSimulationTests(unittest.TestCase):

    # Generated Hawkes process hyper-parameters test function

    @patch("UTILS.utils.write_parquet")
    def test_hyper_params_simulation(self, mock_write_parquet):

        # Called function under test
        params, alpha, beta, mu = hyper_params_simulation()

        # Checked outputs types/shapes 
        self.assertTupleEqual(params.shape, (hwk.PROCESS_NUM, 3))
        self.assertTupleEqual(alpha.shape, (hwk.PROCESS_NUM,))
        self.assertTupleEqual(beta.shape, (hwk.PROCESS_NUM,))
        self.assertTupleEqual(mu.shape, (hwk.PROCESS_NUM,))
        self.assertCountEqual([type(x) for x in (params, alpha, beta, mu)], [np.ndarray] * 4)

        # Checked Parquet file exists
        filename = "hawkes_hyperparams.parquet"
        self.assertTrue(os.path.isfile(filename))

        # Checked Parquet file contents
        df = read_parquet(filename)
        pd.testing.assert_frame_equal(df, pd.DataFrame({"alpha": alpha, "beta": beta, "mu": mu}))

        # Checked write_parquet() was called with correct arguments
        mock_write_parquet.assert_called_once_with({"alpha": alpha, "beta": beta, "mu": mu}, filename=filename)

        # Checked saved Parquet file is not empty
        self.assertGreater(os.path.getsize(self.filename), 0)


    # Simulated Hawkes process test function

    def test_hawkes_simulation(self):

        # Defined parameters, mock and return for Hawkes process
        mock_params = {"mu": 0.1, "alpha": 0.5, "beta": 10.0}
        mock_hawkes_process = Mock()
        mock_hawkes_process.simulate.return_value = np.array([1.0, 2.0, 3.0])

        # Called function under test
        simulator, t = hawkes_simulation(mock_params)

        # Asserted correct kernel, baseline and parameters
        self.assertDictEqual(simulator.__dict__, {'kernel': hwk.KERNEL, 'baseline': hwk.BASELINE, 'params': mock_params})

        # Asserted correct time interval
        mock_hawkes_process.simulate.assert_called_once_with([hwk.TIME_ITV_START, hwk.TIME_HORIZON])

        # Asserted times are the same as mock
        np.testing.assert_array_equal(t, np.array([1.0, 2.0, 3.0]))


    # Simulated several Hawkes processes test function
    
    def test_hawkes_simulations(self):

        # Set up parameters
        alpha = np.array([[0.5]])
        beta = np.array([[1.0]])
        mu = np.array([[0.1]])
        
        # Set up mock return for hawkes_simulation
        hawkes_simulation_mock = Mock(return_value=(np.array([0.5, 0.6]), np.array([1.0, 2.0])))
        
        # Replaced hawkes_simulation with mock using
        with patch.object(HAWKES.hawkes, 'hawkes_simulation', hawkes_simulation_mock):

            # Called function under test
            result = hawkes_simulations(alpha, beta, mu)
            
            # Verified shape/values result
            self.assertEqual(result.shape, (1, 2))
            self.assertTrue(np.array_equal(result, np.array([[1.0, 2.0]])))
        
        # Verify that the mock was called with the correct arguments
        hawkes_simulation_mock.assert_called_once_with({"mu": 0.1, "alpha": 0.5, "beta": 1.0})


    # Estimated Hawkes process test function

    @patch("HAWKES.hawkes.hk.estimator")
    @patch("UTILS.utils.write_parquet")
    def test_hawkes_estimation(self, mock_write_parquet, mock_hk_estimator):

        # Created inputs
        t = np.array([0.2, 0.5, 0.8])
        para = {"mu": 0.5, "alpha": 0.2, "beta": 1.0}
        br, l, aic = 0.5, 10.0, 5.0
        t_trans, interval_trans = np.array([0.0, 0.5, 1.0]), np.array([0.2, 0.3])
        t_pred = np.array([0.3, 0.6, 0.9])

        # Simulated called function in mock
        mock_hawkes_process = MagicMock()
        mock_hawkes_process.para, mock_hawkes_process.br = para, br
        mock_hawkes_process.L, mock_hawkes_process.AIC = l, aic
        mock_hawkes_process.t_trans.return_value = [t_trans, interval_trans]
        mock_hawkes_process.predict.return_value = t_pred
        mock_hk_estimator.return_value = mock_hawkes_process

        # Called function under test
        t_pred_out, metrics_out, t_transform_out, interval_transform_out = hawkes_estimation(t)

        # Checked outputs types/shapes 
        expected_metrics = {"Event(s)": len(t), "Parameters": para, "Branching Ratio": br, "Log-Likelihood": l, "AIC": aic}
        self.assertDictEqual(metrics_out, expected_metrics)

        self.assertTrue(np.allclose([t_pred_out, t_transform_out, interval_transform_out], [t_pred, t_trans, interval_trans]))

        # Checked calls to simulated functions
        mock_hk_estimator.assert_called_once()
        mock_hawkes_process.fit.assert_called_once_with(t, [hwk.TIME_ITV_START, hwk.TIME_HORIZON])
        mock_hawkes_process.t_trans.assert_called_once()
        mock_hawkes_process.predict.assert_called_once_with(hwk.END_T, hwk.NUM_SEQ)
        mock_write_parquet.assert_called_once_with(metrics_out, filename="hawkes_estimation.parquet")

if __name__ == '__main__':
    unittest.main()
