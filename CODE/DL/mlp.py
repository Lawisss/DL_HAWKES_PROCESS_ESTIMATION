#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""MLP module

File containing MLP Aggregated/Binned Hawkes Process estimation.

"""

import os 
from typing import Tuple, Union

import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from torchinfo import summary
from torch.linalg import norm
from torch.utils.data import DataLoader
from torch.nn.utils import parameters_to_vector
from torch.utils.tensorboard import SummaryWriter

import VARIABLES.mlp_var as mlp
from UTILS.utils import profiling
import VARIABLES.evaluation_var as eval
import VARIABLES.preprocessing_var as prep



# MLP creation

class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        # Parameters initialization

        # Created linear layers (first layer = input_size / hidden layers = hidden_size neurons) 
        # * operator unpacked list comprehension into individual layers added to nn.ModuleList
        self.layers = nn.ModuleList([nn.Linear(mlp.INPUT_SIZE, mlp.HIDDEN_SIZE), 
                                     *(nn.Linear(mlp.HIDDEN_SIZE, mlp.HIDDEN_SIZE) for _ in range(mlp.NUM_HIDDEN_LAYERS - 1))])
        
        self.output_layer = nn.Linear(mlp.HIDDEN_SIZE, mlp.OUTPUT_SIZE)
        self.relu = nn.ReLU()
        self.l2_reg = mlp.L2_REG

    # Spread inputs through hidden layers, ReLU function and returns outputs
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        for layer in self.layers:
            x = self.relu(layer(x))
        x = self.output_layer(x)

        return x


# MLP Training

class MLPTrainer(MLP):
    def __init__(self):
        super().__init__()

        # Parameters initialization
        # MLP parameters
        self.model = MLP().to(prep.DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=mlp.LEARNING_RATE)
        self.criterion = nn.MSELoss().to(prep.DEVICE)

        # One epoch train/val loss parameters
        self.train_loss = 0
        self.val_loss = 0

        # Training train/val losses parameters (Many epochs)
        self.train_losses = np.zeros(mlp.MAX_EPOCHS, dtype=np.float32)
        self.val_losses = np.zeros(mlp.MAX_EPOCHS, dtype=np.float32)


    # Sum-up model function

    def summary_model(self) -> str:

        summary(self.model, input_size=mlp.INPUT_SIZE, input_data=[mlp.BATCH_SIZE, mlp.INPUT_SIZE], batch_dim=mlp.BATCH_SIZE, 
                col_names=mlp.SUMMARY_COL_NAMES, device=mlp.DEVICE, mode=mlp.SUMMARY_MODE, verbose=mlp.SUMMARY_VERBOSE)
        
        return f"{mlp.SUMMARY_MODEL:^30} Summary"
    

    # One run function (automatic gradients backpropagation)

    @torch.enable_grad()
    def run_epoch(self, train_loader: DataLoader) -> float:
        
        self.model.train()

        for x, y in train_loader:

            # Forward pass (Autograd)
            self.optimizer.zero_grad()
            output = self.model(x)
            loss = self.criterion(output, y) + (mlp.L2_REG * norm(parameters_to_vector(self.model.parameters()), ord=2, dtype=torch.float32))

            # Backward pass (Autograd)
            loss.backward()
            self.optimizer.step()
            self.train_loss += loss.detach().item() * x.size(0)

        self.train_loss /= len(train_loader.dataset)

        return self.train_loss


    # Evaluation function (Deactivated gradient calculations to speed up evaluation)

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> float:

        self.model.eval()
        
        for x, y in val_loader:
            
            output = self.model(x)
            loss = self.criterion(output, y)
            self.val_loss += loss.item() * x.size(0)

        self.val_loss /= len(val_loader.dataset)

        return self.val_loss


    # Early stopping function (checkpoint)

    def early_stopping(self, best_loss: Union[float, int] = float('inf'), no_improve_count: int = 0) -> bool:

        # Checked early stopping
        # Save model if val_loss has decreased
        if self.val_loss < best_loss:

            torch.save(self.model.state_dict(), f"{os.path.join(prep.FILEPATH, mlp.FILENAME_BEST_MODEL)}")
            best_loss = self.val_loss
            no_improve_count = 0

        else:
            no_improve_count += 1

        # Early stopping condition
        if no_improve_count >= mlp.EARLY_STOP_PATIENCE:
            print(f"Early stopping, no val_loss improvement for {mlp.EARLY_STOP_PATIENCE} epochs")
            return True
        else:
            return False


    # Loading model function (best model)
    def load_model(self) -> str:  
        self.model.load_state_dict(torch.load(f"{os.path.join(prep.FILEPATH, mlp.FILENAME_BEST_MODEL)}"))
        return f"Best model loading ({mlp.FILENAME_BEST_MODEL})..."
    
     
    # Prediction function (estimated Hawkes parameters)
    @torch.no_grad()
    def predict(self, val_X: torch.Tensor, dtype: torch.dtype = torch.float32) -> Tuple[torch.Tensor, float, float]:

        val_Y_pred = self.model(val_X)
        val_eta = torch.mean(val_Y_pred[:, 0], dtype=dtype).item()
        val_mu = torch.mean(val_Y_pred[:, 1], dtype=dtype).item()

        print(f"Validation set - Estimated branching ratio (η): {val_eta:.4f}, Estimated baseline intensity (µ): {val_mu:.4f}")

        return val_Y_pred, val_eta, val_mu


    # Training fonction (PyTorch Profiler = disable)

    @profiling(enable=False)
    def train_model(self, train_loader: DataLoader, val_loader: DataLoader, val_X: torch.Tensor) -> Tuple[nn.Module, np.ndarray, np.ndarray, torch.Tensor, float, float]:

        # Initialized Tensorboard
        writer = SummaryWriter(f"{os.path.join(eval.LOGDIRUN, eval.RUN_NAME)}")

        # Displayed model summary
        print(self.summary_model())

        # Start training
        for epoch in tqdm(range(mlp.MAX_EPOCHS), desc='Training Progress', colour='green'):

            # Converged (Fitted) to optimal parameters
            self.train_losses[epoch] = self.run_epoch(train_loader)

            # Evaluated on validation set
            self.val_losses[epoch] = self.evaluate(val_loader)

            # Checked early stopping
            if self.early_stopping():
                break

            # Updated progress bar description
            tqdm.set_description(f"Epoch {epoch + 1}/{mlp.MAX_EPOCHS} - train_loss: {self.train_losses[epoch]:.4f}, val_loss: {self.val_losses[epoch]:.4f}")

            # Updated progress bar
            tqdm.update(1)

            # Added losses in TensorBoard at each epoch
            writer.add_scalar("Training/Validation Loss", 
                             {'Training': self.train_losses[epoch], 'Validation': self.val_losses[epoch]}, epoch)
            
        # Added model graph to TensorBoard
        writer.add_graph(self.model, val_X)

        # Loaded best model
        print(self.load_model())

        # Computed estimated parameters for validation set (After loaded best model)
        val_Y_pred, val_eta, val_mu = self.predict(val_X)

        # Stored on disk / Closed SummaryWriter
        writer.flush()
        writer.close()

        return self.model, self.train_losses, self.val_losses, val_Y_pred, val_eta, val_mu