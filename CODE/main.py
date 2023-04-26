#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Main module

File containing functions of all executive file

"""

from DL.mlp_model import MLPTrainer
from UTILS.utils import read_parquet, argparser
from HAWKES.hawkes import hawkes_simulations
from HAWKES.discretisation import discretise
from HAWKES.hyperparameters import hyper_params_simulation
from PREPROCESSING.dataset import split_data, create_datasets, create_data_loaders


if __name__ == "__main__":

    args = argparser()

    params, alpha, beta, mu = hyper_params_simulation(filename="test.parquet", args=args)
    simulated_events_seqs = hawkes_simulations(alpha, beta, mu, filename='test.parquet', args=args)
    discret_simulated_events_seqs = discretise(simulated_events_seqs, filename='test.parquet', args=args)

    # x = read_parquet("binned_hawkes_simulations.parquet")
    # y = read_parquet('hawkes_hyperparams.parquet')

    # train_x, train_y, val_x, val_y, test_x, test_y = split_data(x, y.iloc[:, [0, 2]], args=args)
    # train_dataset, val_dataset, test_dataset = create_datasets(train_x, train_y, val_x, val_y, test_x, test_y)
    # train_loader, val_loader, test_loader = create_data_loaders(train_dataset, val_dataset, test_dataset, args=args)
    
    # model, train_losses, val_losses, val_y_pred, val_eta, val_mu = MLPTrainer(args=args).train_model(train_loader, val_loader, val_x)