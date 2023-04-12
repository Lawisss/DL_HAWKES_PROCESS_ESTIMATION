#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""MLP module

File containing MLP Aggregated/Binned Hawkes Process estimation.

"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.linalg import norm
from typing import Tuple, Union
from torch.utils.data import DataLoader
from torch.nn.utils import parameters_to_vector

import VARIABLES.variables as var


# MLP creation
class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_hidden_layers, l2_reg):
        super().__init__()

        # Parameters initialization

        # Created linear layers (first layer = input_size neurons / hidden layers = hidden_size neurons) 
        # * operator unpacked list comprehension into individual layers, which are then added to nn.ModuleList
        self.layers = nn.ModuleList([nn.Linear(input_size, hidden_size), 
                                     *(nn.Linear(hidden_size, hidden_size) for _ in range(num_hidden_layers - 1))])
        
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.l2_reg = l2_reg

    def forward(self, x):

        for layer in self.layers:
            x = self.relu(layer(x))
        x = self.output_layer(x)

        return x


# Building model function

def build_model() -> Tuple[nn.Module, optim.Optimizer, nn.Module]:
    
    # Model/Optimizer/Loss initialization
    model = MLP(var.INPUT_SIZE, var.OUTPUT_SIZE, var.HIDDEN_SIZE, var.NUM_HIDDEN_LAYERS, var.L2_REG).to(var.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=var.LEARNING_RATE)
    criterion = nn.MSELoss().to(var.DEVICE)

    return model, optimizer, criterion


# One run function (automatic gradients backpropagation)

@torch.enable_grad()
def run_epoch(train_loader: DataLoader, model: nn.Module, criterion: nn.Module, optimizer: torch.optim.Optimizer) -> float:
    
    train_loss = 0
    model.train()

    for x, y in train_loader:

        # Forward pass (Autograd)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y) + (var.L2_REG * norm(parameters_to_vector(model.parameters()), ord=2, dtype=torch.float32))

        # Backward pass (Autograd)
        loss.backward()
        optimizer.step()
        train_loss += loss.detach().item() * x.size(0)

    train_loss /= len(train_loader.dataset)

    return train_loss


# Evaluation function (Deactivated gradient calculations to speed up evaluation)

@torch.no_grad()
def evaluate(val_loader: DataLoader, model: nn.Module, criterion: nn.Module) -> float:

    val_loss = 0
    model.eval()
    
    for x, y in val_loader:
        
        output = model(x)
        loss = criterion(output, y)
        val_loss += loss.item() * x.size(0)

    val_loss /= len(val_loader.dataset)

    return val_loss


# Early stopping function (checkpoint)

def early_stopping(model: nn.Module, val_loss: float, best_loss: Union[float, int] = float('inf'), no_improve_count: int = 0, filename: str = "best_model.pt") -> bool:

    # Checked early stopping
    # Save model if val_loss has decreased
    if val_loss < best_loss:

        torch.save(model.state_dict(), f"{var.FILEPATH}{filename}")
        best_loss = val_loss
        no_improve_count = 0

    else:
        no_improve_count += 1

    # Early stopping condition
    if no_improve_count >= var.EARLY_STOP_PATIENCE:
        print(f"Early stopping, no val_loss improvement for {var.EARLY_STOP_PATIENCE} epochs")
        return True
    else:
        return False


# Prediction function (estimated Hawkes parameters)
@torch.no_grad()
def predict(val_X, model, dtype=torch.float32):

    val_Y_pred = model(val_X)
    val_eta = torch.mean(val_Y_pred[:, 0], dtype=dtype).item()
    val_mu = torch.mean(val_Y_pred[:, 1], dtype=dtype).item()
    print(f"Validation set - Estimated branching ratio (η): {val_eta:.4f}, Estimated baseline intensity (µ): {val_mu:.4f}")

    return val_Y_pred, val_eta, val_mu


# Training fonction

def train_model(train_loader, val_loader, val_X, model, criterion, optimizer, filename="best_model.pt"):

    train_loss_list = []
    val_loss_list = []

    for epoch in range(var.MAX_EPOCHS):

        # Converged (Fitted) to optimal parameters
        train_loss = run_epoch(train_loader, model, criterion, optimizer)
        train_loss_list.append(train_loss)

        # Evaluated on validation set
        val_loss = evaluate(val_loader, model, criterion)
        val_loss_list.append(val_loss)

        # Checked early stopping
        if early_stopping(model, val_loss, filename=filename):
            break

        # Training progress
        print(f"Epoch {epoch + 1}/{var.MAX_EPOCHS} - train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}")

    # Loaded best model
    model.load_state_dict(torch.load(f"{var.FILEPATH}{filename}"))

    # Computed estimated parameters for validation set
    val_Y_pred, val_eta, val_mu = predict(val_X, model)

    return model, train_loss, val_loss, val_Y_pred, val_eta, val_mu





# # Training fonction

# def train_model(train_loader, val_loader, model, criterion, optimizer, filename="best_model.pt"):

#     best_loss = float('inf')
#     no_improve_count = 0
#     train_size = len(train_loader.dataset)
#     val_size = len(val_loader.dataset)

#     # Training loop
#     for epoch in range(var.MAX_EPOCHS):

#         # Activated train mode 
#         train_loss = 0
#         model.train()

#         for x, y in train_loader:

#             # Forward pass (Autograd)
#             optimizer.zero_grad()
#             output = model(x)
#             loss = criterion(output, y)
#             loss += var.L2_REG * torch.sum(torch.stack([torch.norm(w, 2) for w in model.parameters()]))

#             # Backward pass (Autograd)
#             loss.backward()
#             optimizer.step()
#             train_loss += loss.detach().item() * x.size(0)

#         train_loss /= train_size

#         # Activated validation mode 
#         val_loss = 0
#         model.eval()

#         # Deactivated gradient calculations to speed up evaluation
#         with torch.no_grad():
#             for x, y in val_loader:

#                 output = model(x)
#                 loss = criterion(output, y)
#                 val_loss += loss.item() * x.size(0)

#             val_loss /= val_size

#         # Checked early stopping
#         # Save model if val_loss has decreased
#         if val_loss < best_loss:
#             torch.save(model.state_dict(), f"{var.FILEPATH}{filename}")
#             best_loss = val_loss
#             no_improve_count = 0
#         else:
#             no_improve_count += 1

#         # Early stopping condition
#         if no_improve_count >= var.EARLY_STOP_PATIENCE:
#             print(f"Early stopping, no val_loss improvement for {var.EARLY_STOP_PATIENCE} epochs")
#             break
        
#         # Training progress
#         print(f"Epoch {epoch + 1}/{var.MAX_EPOCHS} - train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}")

#     # Loaded best model
#     model.load_state_dict(torch.load(f"{var.FILEPATH}{filename}"))

#     # Computed estimated parameters for validation set
#     with torch.no_grad():
#         y_val_pred = model(val_X)
#         val_eta = torch.mean(y_val_pred[:, 0]).item()
#         val_mu = torch.mean(y_val_pred[:, 1]).item()
#         print(f"Validation set - Estimated branching ratio (η): {val_eta:.4f}, Estimated baseline intensity (µ): {val_mu:.4f}")

#     return model


       
