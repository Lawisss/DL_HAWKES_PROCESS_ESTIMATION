#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Main module

File containing functions of all executive file

"""

from DL.mlp_model import MLPTrainer
from PREPROCESSING.dataset import split_data, create_datasets, create_data_loaders


if __name__ == "__name__":

    train_x, train_y, val_x, val_y, test_x, test_y = split_data(x, y)
    train_dataset, val_dataset, test_dataset = create_datasets(train_x, train_y, val_x, val_y, test_x, test_y)
    train_loader, val_loader, test_loader = create_data_loaders(train_dataset, val_dataset, test_dataset)

    model, train_losses, val_losses, val_y_pred, val_eta, val_mu = MLPTrainer().train_model(train_loader, val_loader, val_x)