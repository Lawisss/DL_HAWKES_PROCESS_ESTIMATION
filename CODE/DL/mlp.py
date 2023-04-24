#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""MLP module

File containing MLP Aggregated/Binned Hawkes Process estimation

"""

import os 
from typing import Tuple, Union

import copy
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
from UTILS.utils import profiling, argparser
import VARIABLES.evaluation_var as eval
import VARIABLES.preprocessing_var as prep



# MLP creation
@argparser(parse_args=False, arg_groups=['mlp_params'])
class MLP(nn.Module):
    def __init__(self, args_parsed):
        super().__init__()

        # Parameters initialization
        self.input_size, self.hidden_size, self.num_hidden_layers, self.output_size, self.l2_reg = \
        (args_parsed.input_size, args_parsed.hidden_size, args_parsed.num_hidden_layers, args_parsed.output_size, args_parsed.l2_reg) \
        if args_parsed else (mlp.INPUT_SIZE, mlp.HIDDEN_SIZE, mlp.NUM_HIDDEN_LAYERS, mlp.OUTPUT_SIZE, mlp.L2_REG)

        # Created linear layers (first layer = input_size / hidden layers = hidden_size neurons) 
        # * operator unpacked list comprehension into individual layers added to nn.ModuleList
        self.layers = nn.ModuleList([nn.Linear(self.input_size, self.hidden_size), 
                                     *(nn.Linear(self.hidden_size, self.hidden_size) for _ in range(self.num_hidden_layers - 1))])
        
        self.output_layer = nn.Linear(self.hidden_size, self.output_size)
        self.relu = nn.ReLU()

    # Spread inputs through hidden layers, ReLU function and returns outputs
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        """
        Applied forward pass through NN layers and returned output tensor

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size)
        """

        for layer in self.layers:
            x = self.relu(layer(x))
        x = self.output_layer(x)

        return x


# MLP Training
@argparser(parse_args=False, arg_groups=['device_params', 'mlp_params'])
class MLPTrainer(MLP):
    def __init__(self, args_parsed):
        super().__init__()

        # Parameters initialization
        self.lr, self.max_epochs, self.device = (args_parsed.learning_rate, args_parsed.max_epochs, torch.device(args_parsed.device)) \
                                                if args_parsed else (mlp.LEARNING_RATE, mlp.MAX_EPOCHS, prep.DEVICE)
        
        # MLP parameters
        self.model = MLP().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss().to(self.device)

        # One epoch train/val loss parameters
        self.train_loss = 0
        self.val_loss = 0

        # Training train/val losses parameters (Many epochs)
        self.train_losses = np.zeros(self.max_epochs, dtype=np.float32)
        self.val_losses = np.zeros(self.max_epochs, dtype=np.float32)


    # Sum-up model function

    @argparser(parse_args=False, arg_groups=['loader_params', 'summary_params'])
    def summary_model(self, args_parsed) -> str:

        """
        Return summary of model architecture
        
        Returns:
            str: Summary of model's architecture
        """

        # Parameters initialization
        self.batch_size, self.col_names, self.mode, self.verbose, self.sum_model = (args_parsed.batch_size, args_parsed.summary_col_names, args_parsed.summary_mode, args_parsed.summary_verbose, args_parsed.summary_model) \
                                                                   if args_parsed else (prep.BATCH_SIZE, mlp.SUMMARY_COL_NAMES, mlp.SUMMARY_MODE, mlp.SUMMARY_VERBOSE, mlp.SUMMARY_MODEL)

        summary(self.model, input_size=self.input_size, input_data=[self.batch_size, self.input_size], 
                batch_dim=self.batch_size, col_names=self.col_names, device=self.device, mode=self.mode, verbose=self.verbose)
        
        return f"{self.sum_model:^30} Summary"
    

    # One run function (automatic gradients backpropagation)

    @torch.enable_grad()
    def run_epoch(self, train_loader: DataLoader) -> float:

        """
        Run one training epoch and returned average training loss

        Args:
            train_loader (DataLoader): Training set

        Returns:
            float: Average training loss over one epoch
        """
        
        self.model.train()

        for x, y in train_loader:

            # Forward pass (Autograd)
            self.optimizer.zero_grad()
            output = self.model(x)
            loss = self.criterion(output, y) + (self.l2_reg * norm(parameters_to_vector(self.model.parameters()), ord=2, dtype=torch.float32))

            # Backward pass (Autograd)
            loss.backward()
            self.optimizer.step()
            self.train_loss += loss.detach().item() * x.size(0)

        self.train_loss /= len(train_loader.dataset)

        return self.train_loss


    # Evaluation function (Deactivated gradient calculations to speed up evaluation)
    
    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> float:

        """
        Evaluated model performance on validation set

        Args:
            val_loader (DataLoader): Validation set

        Returns:
            float: Average validation loss over validation set
        """

        self.model.eval()
        
        for x, y in val_loader:
            
            output = self.model(x)
            loss = self.criterion(output, y)
            self.val_loss += loss.item() * x.size(0)

        self.val_loss /= len(val_loader.dataset)

        return self.val_loss


    # Early stopping function (checkpoint)

    @argparser(parse_args=False, arg_groups=['file_params', 'mlp_params'])
    def early_stopping(self, args_parsed, best_loss: Union[float, int] = float('inf'), no_improve_count: int = 0) -> bool:
        
        """
        Checked early stopping condition based on validation loss

        Args:
            best_loss (float or int, optional): Current best validation loss (default: float('inf'))
            no_improve_count (int, optional): Number of epochs with no improvement in validation loss (default: 0)

        Returns:
            bool: True if early stopping condition is met, False otherwise
        """    

        # Parameters initialization
        self.filepath, self.filename_best_model, self.early_stop_patience = (args_parsed.filepath, args_parsed.filename_best_model, args_parsed.early_stop_patience) \
                                                                            if args_parsed else (prep.FILEPATH, mlp.FILENAME_BEST_MODEL, mlp.EARLY_STOP_PATIENCE)

        
        # Checked early stopping
        # Save model if val_loss has decreased
        if self.val_loss < best_loss:
            
            torch.save(copy.deepcopy(self.model.state_dict()), f"{os.path.join(self.filepath, self.filename_best_model)}")
            best_loss = self.val_loss
            no_improve_count = 0

        else:
            no_improve_count += 1

        # Early stopping condition
        if no_improve_count >= self.early_stop_patience:
            print(f"Early stopping, no val_loss improvement for {self.early_stop_patience} epochs")
            return True
        else:
            return False


    # Loading model function (best model)

    def load_model(self) -> str:  

        """
        Loaded best model from saved file

        Returns:
            str: Message indicating that best model has been loaded
        """     

        self.model.load_state_dict(torch.load(f"{os.path.join(self.filepath, self.filename_best_model)}"))
        return f"Best model loading ({self.filename_best_model})..."
    
     
    # Prediction function (estimated Hawkes parameters)

    @torch.no_grad()
    def predict(self, val_x: torch.Tensor, dtype: torch.dtype = torch.float32) -> Tuple[torch.Tensor, float, float]:

        """
        Estimated Hawkes parameters using validation dataset

        Args:
            val_x (torch.Tensor): Input tensor for validation dataset
            dtype (torch.dtype): Type for tensor operations (default: torch.float32)

        Returns:
            Tuple[torch.Tensor, float, float]: Estimations for branching ratio (eta), and baseline intensity (mu)
        """        

        val_y_pred = self.model(val_x)
        val_eta = torch.mean(val_y_pred[:, 0], dtype=dtype).item()
        val_mu = torch.mean(val_y_pred[:, 1], dtype=dtype).item()

        print(f"Validation set - Estimated branching ratio (η): {val_eta:.4f}, Estimated baseline intensity (µ): {val_mu:.4f}")

        return val_y_pred, val_eta, val_mu


    # Training fonction (PyTorch Profiler = disable)
    
    @profiling(enable=False)
    def train_model(self, args_parsed, train_loader: DataLoader, val_loader: DataLoader, val_x: torch.Tensor) -> Tuple[nn.Module, np.ndarray, np.ndarray, torch.Tensor, float, float]:

        """
        Trained and evaluated model

        Args:
            train_loader (DataLoader): Training data
            val_loader (DataLoader): Validation data
            val_x (torch.Tensor): Input features for validation data

        Returns:
            Tuple[nn.Module, np.ndarray, np.ndarray, torch.Tensor, float, float]: Model, losses, predictions, eta/mu
        """

        # Initialized Tensorboard
        writer = SummaryWriter(f"{os.path.join(eval.LOGDIRUN, eval.RUN_NAME)}")

        # Displayed model summary
        print(self.summary_model())

        # Start training
        for epoch in tqdm(range(self.max_epochs), desc='Training Progress', colour='green'):

            # Converged (Fitted) to optimal parameters
            self.train_losses[epoch] = self.run_epoch(train_loader)

            # Evaluated on validation set
            self.val_losses[epoch] = self.evaluate(val_loader)

            # Checked early stopping
            if self.early_stopping():
                break

            # Updated progress bar description
            tqdm.set_description(f"Epoch {epoch + 1}/{self.max_epochs} - train_loss: {self.train_losses[epoch]:.4f}, val_loss: {self.val_losses[epoch]:.4f}")

            # Updated progress bar
            tqdm.update(1)

            # Added losses in TensorBoard at each epoch
            writer.add_scalar("Training/Validation Loss", 
                             {'Training': self.train_losses[epoch], 'Validation': self.val_losses[epoch]}, epoch)
            
        # Added model graph to TensorBoard
        writer.add_graph(self.model, val_x)

        # Loaded best model
        print(self.load_model())

        # Computed estimated parameters for validation set (After loaded best model)
        val_y_pred, val_eta, val_mu = self.predict(val_x)

        # Stored on disk / Closed SummaryWriter
        writer.flush()
        writer.close()

        return self.model, self.train_losses, self.val_losses, val_y_pred, val_eta, val_mu