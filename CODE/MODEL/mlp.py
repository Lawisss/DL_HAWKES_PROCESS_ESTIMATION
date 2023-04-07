#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""MLP module

File containing MLP Aggregated/Binned Hawkes Process estimation.

"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

import VARIABLES.variables as var

# Input data
X = np.random.randn(var.INPUT_SIZE, 1).astype(np.float32)
Y = np.random.randn(var.INPUT_SIZE, var.OUTPUT_SIZE).astype(np.float32)

# Train/Val/Test split

val_size = int(len(X) * var.VAL_RATIO)
test_size = int(len(X) * var.TEST_RATIO)
train_size = len(X) - val_size - test_size

train_X, val_X, test_X = torch.tensor(X[:train_size], dtype=torch.float32), torch.tensor(X[train_size:train_size+val_size], dtype=torch.float32), torch.tensor(X[train_size+val_size:], dtype=torch.float32)
train_Y, val_Y, test_Y = torch.tensor(Y[:train_size], dtype=torch.float32), torch.tensor(Y[train_size:train_size+val_size], dtype=torch.float32), torch.tensor(Y[train_size+val_size:], dtype=torch.float32)

# Creation of datasets
train_dataset = TensorDataset(train_X, train_Y)
val_dataset = TensorDataset(val_X, val_Y)
test_dataset = TensorDataset(test_X, test_Y)

# Creation of data loaders
train_loader = DataLoader(train_dataset, batch_size=var.BATCH_SIZE, shuffle=True) # num_workers=4, pin_memory=True
val_loader = DataLoader(val_dataset, batch_size=var.BATCH_SIZE) # num_workers=4, pin_memory=True
test_loader = DataLoader(test_dataset, batch_size=var.BATCH_SIZE) # num_workers=4, pin_memory=True

# MLP Creation
class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_hidden_layers, l2_reg):
        super().__init__()

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_size))

        for _ in range(num_hidden_layers - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))

        self.output_layer = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.l2_reg = l2_reg

    def forward(self, x):

        for layer in self.layers:
            x = self.relu(layer(x))
        x = self.output_layer(x)

        return x # .cuda()
    
# Model/Optimizer initialization
model = MLP(var.INPUT_SIZE, var.OUTPUT_SIZE, var.HIDDEN_SIZE, var.NUM_HIDDEN_LAYERS, var.L2_REG) # .cuda()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Loss function
criterion = nn.MSELoss()

def train_model(train_loader, val_loader, model, criterion, optimizer, var):

    best_loss = float('inf')
    no_improve_count = 0
    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset)

    # Training loop
    for _ in range(var.MAX_EPOCHS):

        train_loss = 0
        model.train()

        for x, y in train_loader:

            optimizer.zero_grad()
            output = model(x) # .cuda()
            loss = criterion(output, y) # .cuda()
            l2_loss = sum(w.norm(2) for w in model.parameters())
            loss += var.L2_REG * l2_loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.shape[0]

        train_loss /= train_size

        val_loss = 0
        model.eval()

        with torch.no_grad():
            for x, y in val_loader:

                output = model(x) # .cuda()
                loss = criterion(output, y) # .cuda()
                val_loss += loss.item() * x.shape[0]

            val_loss /= val_size

        # Checked whether early stopping should be applied
        # Save model if validation loss has decreased
        if val_loss < best_loss:
            torch.save(model.state_dict(), 'best_model.pt')
            best_loss = val_loss
            no_improve_count = 0
        else:
            no_improve_count += 1

        # Early stopping condition
        if no_improve_count >= var.EARLY_STOP_PATIENCE:
            print(f"Early stopping, no val_loss improvement for {var.EARLY_STOP_PATIENCE} epochs")
            break

    # Loaded best model
    model.load_state_dict(torch.load('best_model.pt'))

    # Evaluated model
    model.eval()

    # Computed estimated parameters for the validation set
    with torch.no_grad():
        y_val_pred = model(val_X)
        val_eta = y_val_pred[:, 0].mean().item()
        val_mu = y_val_pred[:, 1].mean().item()
        print(f"Validation set - Estimated branching ratio (η): {val_eta:.4f}, Estimated baseline intensity (µ): {val_mu:.4f}")

    return model


       
