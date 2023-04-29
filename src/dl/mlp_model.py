#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""MLP module

File containing MLP Aggregated/Binned Hawkes Process estimation (eta/mu)

"""

import os 
import copy
from typing import Tuple, Union, Optional, Callable

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

import variables.mlp_var as mlp
from tools.utils import profiling, write_parquet
import variables.eval_var as eval
import variables.prep_var as prep


# MLP creation

class MLP(nn.Module):
    def __init__(self, input_size=None, hidden_size=None, num_hidden_layers=None, output_size=None):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.output_size = output_size

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

class MLPTrainer:
    def __init__(self, args: Optional[Callable] = None):

        # Initialized parameters
        params = [('input_size', mlp.INPUT_SIZE),
                  ('hidden_size', mlp.HIDDEN_SIZE),
                  ('num_hidden_layers', mlp.NUM_HIDDEN_LAYERS),
                  ('output_size', mlp.OUTPUT_SIZE),
                  ('device', prep.DEVICE),
                  ('learning_rate', mlp.LEARNING_RATE),
                  ('max_epochs', mlp.MAX_EPOCHS),
                  ('l2_reg', mlp.L2_REG),
                  ('batch_size', prep.BATCH_SIZE),
                  ('summary_col_names', mlp.SUMMARY_COL_NAMES),
                  ('summary_mode', mlp.SUMMARY_MODE),
                  ('summary_verbose', mlp.SUMMARY_VERBOSE),
                  ('sumup_model', mlp.SUMMARY_MODEL),
                  ('early_stop_delta', mlp.EARLY_STOP_DELTA),
                  ('dirpath', prep.DIRPATH),
                  ('filename_best_model', mlp.FILENAME_BEST_MODEL),
                  ('early_stop_patience', mlp.EARLY_STOP_PATIENCE),
                  ('logdirun', eval.LOGDIRUN),
                  ('run_name', eval.RUN_NAME)]
        
        for attr, default_val in params:
            setattr(self, attr, getattr(args, attr, default_val))

        # MLP parameters
        self.model = MLP(self.input_size, self.hidden_size, self.num_hidden_layers, self.output_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss().to(self.device)

        # One epoch train/val loss parameters
        self.train_loss = 0
        self.val_loss = 0

        # Test loss/predictions parameters
        self.test_loss = 0
        self.test_eta_pred = 0
        self.test_mu_pred = 0
        
        # Training train/val losses parameters (Many epochs)
        self.train_losses = np.zeros(self.max_epochs, dtype=np.float32)
        self.val_losses = np.zeros(self.max_epochs, dtype=np.float32)

    # Sum-up model function

    def summary_model(self) -> str:

        """
        Return summary of model architecture

        Returns:
            str: Summary of model's architecture
        """

        summary(self.model, input_size=(self.batch_size, self.input_size), col_names=self.summary_col_names, 
                device=self.device, mode=self.summary_mode, verbose=self.summary_verbose)
        
        return f"{self.sumup_model:^30} Summary"
    

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

    def early_stopping(self, best_loss: Union[float, int] = float('inf'), no_improve_count: int = 0) -> bool:
        
        """
        Checked early stopping condition based on validation loss

        Args:
            best_loss (float or int, optional): Current best validation loss (default: float('inf'))
            no_improve_count (int, optional): Number of epochs with no improvement in validation loss (default: 0)

        Returns:
            bool: True if early stopping condition is met, False otherwise
        """

        # Checked early stopping
        # Save model if val_loss has decreased
        if (self.val_loss + self.early_stop_delta) < best_loss:
            
            torch.save(copy.deepcopy(self.model.state_dict()), os.path.join(self.dirpath, self.best_model_dir, self.filename_best_model))
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

        Args:
            folder (str, optional): Sub-folder name in results folder (default: 'best_model')

        Returns:
            str: Message indicating that best model has been loaded
        """     

        self.model.load_state_dict(torch.load(os.path.join(self.dirpath, self.best_model_dir, self.filename_best_model)))
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
        val_eta_pred = torch.mean(val_y_pred[:, 0], dtype=dtype).item()
        val_mu_pred = torch.mean(val_y_pred[:, 1], dtype=dtype).item()

        print(f"Validation set - Estimated branching ratio (η): {val_eta_pred:.4f}, Estimated baseline intensity (µ): {val_mu_pred:.4f}")

        return val_y_pred, val_eta_pred, val_mu_pred


    # Training fonction (PyTorch Profiler = disable)
    
    @profiling(enable=False)
    def train_model(self, train_loader: DataLoader, val_loader: DataLoader, val_x: torch.Tensor, val_y: torch.Tensor) -> Tuple[nn.Module, np.ndarray, np.ndarray, torch.Tensor, float, float]:

        """
        Trained and evaluated model

        Args:
            train_loader (DataLoader): Training data
            val_loader (DataLoader): Validation data
            val_x (torch.Tensor): Input features for validation data
            val_y (torch.Tensor): Label outputs for validation data

        Returns:
            Tuple[nn.Module, np.ndarray, np.ndarray, torch.Tensor, float, float]: Model, losses, predictions, eta/mu
        """

        # Initialized Tensorboard
        writer = SummaryWriter(os.path.join(self.logdirun, self.train_dir, self.run_name))

        # Displayed model summary
        print(self.summary_model())

        # Start training
        with tqdm(total=self.max_epochs, desc='Training Progress', colour='green') as pbar:
            
            for epoch in range(self.max_epochs):

                # Converged (Fitted) to optimal parameters
                self.train_losses[epoch] = self.run_epoch(train_loader)

                # Evaluated on validation set
                self.val_losses[epoch] = self.evaluate(val_loader)

                # Checked early stopping
                if self.early_stopping():
                    break

                # Updated progress bar description
                pbar.set_description(f"Epoch {epoch + 1}/{self.max_epochs} - train_loss: {self.train_losses[epoch]:.4f}, val_loss: {self.val_losses[epoch]:.4f}")

                # Updated progress bar
                pbar.update(1)

                # Added losses in TensorBoard at each epoch
                writer.add_scalars("Loss", {"Training": self.train_losses[epoch], "Validation": self.val_losses[epoch]}, epoch)

        # Added model graph to TensorBoard
        writer.add_graph(self.model, val_x)

        # Loaded best model
        print(self.load_model())

        # Computed estimated parameters for validation set (After loaded best model)
        val_y_pred, val_eta_pred, val_mu_pred = self.predict(val_x)

        # Added results histograms to TensorBoard
        writer.add_histogram("Baseline intensity Histogram", val_y_pred[:, 1], len(val_y), bins="auto")
        writer.add_histogram("Branching ratio Histogram", val_y_pred[:, 0], len(val_y), bins="auto")
        writer.add_histogram("Prediction Histogram", val_y_pred, len(val_y), bins="auto")
        writer.add_histogram("Validation Histogram", val_y, len(val_y), bins="auto")

        # Stored on disk / Closed SummaryWriter
        writer.flush()
        writer.close()
        
        # Written parameters to Parquet file
        write_parquet({'train_losses': self.train_losses, 
                       'val_losses': self.val_losses}, 
                       filename=f"{self.run_name}_losses.parquet", 
                       folder=os.path.join(self.logdirun, self.train_dir, self.run_name))
        
        write_parquet({'val_eta_true': val_y[:, 0], 
                       'val_mu_true': val_y[:, 1],
                       'val_eta_pred': val_y_pred[:, 0], 
                       'val_mu_pred': val_y_pred[:, 1]}, 
                       filename=f"{self.run_name}_predictions.parquet", 
                       folder=os.path.join(self.logdirun, self.train_dir, self.run_name))

        return self.model, self.train_losses, self.val_losses, val_y_pred, val_eta_pred, val_mu_pred
    

    # Testing fonction (PyTorch Profiler = disable)
    
    @profiling(enable=False)
    @torch.no_grad()
    def test_model(self, test_loader: DataLoader, test_y: torch.Tensor, dtype: torch.dtype = torch.float32) -> Tuple[np.ndarray, float, float, float]:

        """
        Tested and evaluated model

        Args:
            test_loader (DataLoader): Testing data
            test_y (torch.Tensor): Label outputs for testing data
            dtype (torch.dtype): Predictions datatype (default: torch.float32)

        Returns:
            Tuple[np.ndarray, float, float, float]: predictions, loss average, eta average, mu average
        """
        
        # Eval mode
        self.model.eval()

        # Initialized parameters
        index = 0
        test_y_pred = torch.empty((len(test_loader.dataset), 2), dtype=dtype)
        
        # Initialized Tensorboard
        writer = SummaryWriter(os.path.join(self.logdirun, self.test_dir, self.run_name))

        for x, y in test_loader:
            
            # Forward pass for predictions
            y_pred = self.model(x)
            test_y_pred[index:index+len(x), :] = y_pred
            index += len(x)

            # Computed loss
            loss = self.criterion(y_pred, y)
            self.test_loss += loss.item() * x.size(0)

            # Computed branching ratio and baseline intensity predictions
            self.test_eta_pred += torch.mean(y_pred[:, 0], dtype=dtype).item() * x.size(0)
            self.test_mu_pred += torch.mean(y_pred[:, 1], dtype=dtype).item() * x.size(0)

            # Add losses, eta, mu values in TensorBoard
            writer.add_scalars("Prediction", {"Branching ratio": self.test_eta_pred, "Baseline intensity": self.test_mu_pred}, index)
            writer.add_scalars("Loss", {"Test Loss": self.test_loss}, index)

            # Added results histograms to TensorBoard
            writer.add_histogram("Baseline intensity Histogram", test_y_pred[:, 1][index], index, bins="auto")
            writer.add_histogram("Branching ratio Histogram", test_y_pred[:, 0][index], index, bins="auto")
            writer.add_histogram("Prediction Histogram", test_y_pred[index], index, bins="auto")
            writer.add_histogram("Validation Histogram", test_y[index], index, bins="auto")

        # Computed average loss and predictions
        self.test_loss /= len(test_loader.dataset)
        self.test_eta_pred /= len(test_loader.dataset)
        self.test_mu_pred /= len(test_loader.dataset)

        print(f"Test set - Test loss: {self.test_loss:.4f}, Estimated branching ratio (η): {self.test_eta_pred:.4f}, Estimated baseline intensity (µ): {self.test_mu_pred:.4f}")

        # Stored on disk / Closed SummaryWriter
        writer.flush()
        writer.close()

        # Written parameters to parquet file
        write_parquet({'test_eta_true': test_y[:, 0], 
                       'test_mu_true': test_y[:, 1],
                       'test_eta_pred': test_y_pred[:, 0], 
                       'test_mu_pred': test_y_pred[:, 1]}, 
                       filename=f"{self.run_name}_predictions.parquet", 
                       folder=os.path.join(self.logdirun, self.test_dir, self.run_name))

        return test_y_pred, self.test_loss, self.test_eta_pred, self.test_mu_pred