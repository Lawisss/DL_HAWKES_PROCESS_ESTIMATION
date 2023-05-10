#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Main module

File containing executive function

"""

import argparse
from typing import Optional

from dl.mlp_model import MLPTrainer
from dl.linear_model import linear_model
from tools.utils import read_parquet
from hawkes.simulation import hawkes_simulations
from hawkes.discretisation import discretise
from hawkes.hyperparameters import hyper_params_simulation
from preprocessing.dataset import split_data, create_datasets, create_data_loaders


# Main function

def main(args: Optional[argparse.Namespace] = None) -> None:

    """
    Project main executive function

    Args:
        args (argparse.Namespace, optional): CLI arguments storing (default: None)

    Returns:
        None: Function does not return anything

    """

    params, alpha, beta, eta, mu = hyper_params_simulation(filename="hawkes_hyperparams.parquet", args=args)
    simulated_events_seqs = hawkes_simulations(alpha, beta, mu, filename='hawkes_simulations.parquet', args=args)
    discret_simulated_events_seqs = discretise(simulated_events_seqs, filename='binned_hawkes_simulations.parquet', args=args)

    train_x, train_y, val_x, val_y, test_x, test_y = split_data(discret_simulated_events_seqs, params[:, [2, 3]], args=args)
    train_dataset, val_dataset, test_dataset = create_datasets(train_x, train_y, val_x, val_y, test_x, test_y)
    train_loader, val_loader, test_loader = create_data_loaders(train_dataset, val_dataset, test_dataset, args=args)

    model, train_losses, val_losses, val_y_pred, val_eta, val_mu = MLPTrainer(args=args).train_model(train_loader, val_loader, val_x, val_y)
    test_y_pred, test_eta_pred, test_mu_pred = MLPTrainer().test_model(test_x, test_y)
    
    


    

    