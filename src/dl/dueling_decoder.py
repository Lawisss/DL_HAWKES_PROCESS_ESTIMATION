#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""VAE with dueling decoder module

File containing VAE conditional intensities and parameters estimation (lambda, eta/mu)

"""

import os
import io
import copy
from typing import Tuple, Optional, Callable

import torch
# import stan
import numpy as np
import polars as pl
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from torchinfo import summary
from torch.nn.functional import poisson_nll_loss
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter

from tools.utils import profiling, write_parquet
import variables.vae_var as vae
import variables.eval_var as eval
import variables.prep_var as prep


# Poisson VAE with dueling decoder

class PoissonVAE(nn.Module):
    def __init__(self, input_size: Optional[int] = None, latent_size: Optional[int] = None, intermediate_size: Optional[int] = None):
        super().__init__()

        # Initialized parameters
        self.input_size = input_size
        self.latent_size = latent_size
        self.intermediate_size = intermediate_size

        # Initialized encoder
        self.encoder = nn.Sequential(nn.Linear(input_size, intermediate_size), 
                                     nn.ReLU(),
                                     nn.Linear(intermediate_size, int(intermediate_size * 0.5)),
                                     nn.ReLU())

        self.latent_mean = nn.Linear(int(intermediate_size * 0.5), latent_size)
        self.latent_log_var = nn.Linear(int(intermediate_size * 0.5), latent_size)

        # Initialized decoders
        self.intensities_decoder = nn.Sequential(nn.Linear(latent_size, int(intermediate_size * 0.5)),
                                                 nn.ReLU(),
                                                 nn.Linear(int(intermediate_size * 0.5), intermediate_size),
                                                 nn.ReLU(),
                                                 nn.Linear(intermediate_size, input_size),
                                                 nn.Softplus())

        self.parameters_decoder = nn.Sequential(nn.Linear(latent_size, 15),
                                                nn.ReLU(),
                                                nn.Linear(15, 15),
                                                nn.ReLU(),
                                                nn.Linear(15, 2),
                                                nn.Softplus())

    # Reparameterized function

    def reparameterize(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:

        """
        Reparameterization trick for sampling from Gaussian distribution

        Args:
            mean (torch.Tensor): Mean of Gaussian distribution
            log_var (torch.Tensor): Log variance of Gaussian distribution

        Returns:
            torch.Tensor: Sampled latent variable 
        """

        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)

        return mean + (std * epsilon)
    

    # Encoded function

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        """
        Encode input data into latent space

        Args:
            x (torch.Tensor): Input data

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Encoded mean and log variance
        """

        h = self.encoder(x)
        mean = self.latent_mean(h)
        log_var = self.latent_log_var(h)

        return mean, log_var


    # Decoded intensities function

    def decode_intensities(self, z):

        """
        Decode latent variable into reconstructed intensities

        Args:
            z (torch.Tensor): Latent variable

        Returns:
            torch.Tensor: Reconstructed data
        """

        intensities_pred = self.intensities_decoder(z)

        return intensities_pred


    # Decoded parameters function

    def decode_parameters(self, z):

        """
        Decode latent variable into reconstructed parameters (eta/mu)

        Args:
            z (torch.Tensor): Latent variable

        Returns:
            torch.Tensor: Reconstructed data
        """

        params_pred = self.parameters_decoder(z)

        return params_pred
    

    # Spread inputs through encoding/decoding

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        """
        Poisson VAE with dueling decoder forward pass

        Args:
            x (torch.Tensor): Input data

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Reconstructed x, mean, and log variance
        """

        mean, log_var = self.encode(x)
        latent = self.reparameterize(mean, log_var)
        intensities = self.decode_intensities(latent)
        parameters = self.decode_parameters(latent)
        y_pred = torch.cat((intensities, parameters), dim=1)

        return y_pred, mean, log_var


class DuelingTrainer:
    def __init__(self, args: Optional[Callable] = None):

        # Initialized parameters
        params = [('input_size', vae.INPUT_SIZE),
                  ('latent_size', vae.LATENT_SIZE),
                  ('intermediate_size', vae.INTERMEDIATE_SIZE),
                  ('device', prep.DEVICE),
                  ('weight_decay', vae.WEIGHT_DECAY),
                  ('learning_rate', vae.LEARNING_RATE),
                  ('max_epochs', vae.MAX_EPOCHS),
                  ('batch_size', prep.BATCH_SIZE),
                  ('kl_start', vae.KL_START),
                  ('kl_steep', vae.KL_STEEP),
                  ('anneal_target', vae.ANNEAL_TARGET),
                  ('min_cycles', vae.MIN_CYCLES),
                  ('summary_col_names', vae.SUMMARY_COL_NAMES),
                  ('summary_mode', vae.SUMMARY_MODE),
                  ('summary_verbose', vae.SUMMARY_VERBOSE),
                  ('sumup_model', vae.SUMMARY_MODEL),
                  ('dirpath', prep.DIRPATH),
                  ('filename_best_model', vae.FILENAME_BEST_MODEL),
                  ('logdirun', eval.LOGDIRUN),
                  ('train_dir', eval.TRAIN_DIR),
                  ('test_dir', eval.TEST_DIR),
                  ('best_model_dir', eval.BEST_MODEL_DIR),
                  ('run_name', eval.RUN_NAME)]
        
        for attr, default_val in params:
            setattr(self, attr, getattr(args, attr, default_val))
    
        # VAE parameters
        self.model = PoissonVAE(self.input_size, self.latent_size, self.intermediate_size).to(self.device)

        optimizer_params = [{'params': self.model.encoder.parameters()}, 
                            {'params': self.model.latent_mean.parameters()},
                            {'params': self.model.latent_log_var.parameters()}, 
                            {'params': self.model.intensities_decoder.parameters()}]

        for idx, layer in enumerate(self.model.parameters_decoder):
            if idx == 0 or idx == 2:
                optimizer_params.append({'params': layer.parameters(), 'weight_decay': self.weight_decay})
            else:
                optimizer_params.append({'params': layer.parameters()})
        
        self.optimizer = optim.Adam(optimizer_params, lr=self.learning_rate)
        self.weight = torch.tensor(0.0, dtype=torch.float32)
        self.cycle = torch.tensor(1.0, dtype=torch.float32)

        # One epoch train/val loss parameters
        self.train_loss = 0
        self.val_loss = 0

        # Training train/val losses parameters (Many epochs)
        self.train_losses = np.zeros(self.max_epochs, dtype=np.float32)
        self.val_losses = np.zeros(self.max_epochs, dtype=np.float32)
    

    # VAE loss function

    def vae_loss(self, y: torch.Tensor, y_pred: torch.Tensor, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:

        """
        Compute VAE loss

        Args:
            y (torch.Tensor): Input data
            y_pred (torch.Tensor): Reconstructed data
            mean (torch.Tensor): Mean of latent distribution
            log_var (torch.Tensor): Log variance of latent distribution

        Returns:
            torch.Tensor: VAE loss
        """

        param_loss = nn.MSELoss()(y_pred[:, self.input_size:(self.input_size + 2)], y[:, self.input_size:(self.input_size + 2)])
        recon_loss = torch.sum(poisson_nll_loss(y_pred[:, :self.input_size], y[:, :self.input_size], reduction='none'), dim=-1, dtype=torch.float32)
        kl_loss = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=-1, dtype=torch.float32)

        return torch.mean(((1 / (1 + self.weight)) * recon_loss) + (250 * param_loss) + (self.weight * kl_loss), dtype=torch.float32)


    # Sum-up model function

    def summary_model(self) -> str:

        """
        Summary of model architecture

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
        
        # Training mode
        self.model.train()

        for x, y in train_loader:

            # Forward pass (Autograd)
            self.optimizer.zero_grad()
            y_pred, mean, log_var = self.model(x)
            loss = self.vae_loss(y, y_pred, mean, log_var)
            
            # Backward pass (Autograd)
            loss.backward()

            # Gradient clipping
            clip_grad_norm_(self.model.parameters(), 1000)
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

        # Evaluation mode
        self.model.eval()
        
        for x, y in val_loader:
            
            y_pred, mean, log_var = self.model(x)
            loss = self.vae_loss(y, y_pred, mean, log_var)
            self.val_loss += loss.item() * x.size(0)

        self.val_loss /= len(val_loader.dataset)

        return self.val_loss


    # Cyclical Annealing function (checkpoint)

    def cyclical_annealing(self, epoch: int) -> None:

        """
        Apply cyclical annealing to KL divergence weight
        
        Args:
            epoch (int): Current epoch number

        Returns:
            None: Function does not return anything
        
        """
        
        # Retrieve current cycle and target multiplier
        current_cycle = self.cycle.item()
        target_mult = min(current_cycle / self.min_cycles, 1)

        # Save best model
        torch.save(copy.deepcopy(self.model.state_dict()), os.path.join(self.dirpath, self.best_model_dir, self.filename_best_model))

        if epoch >= self.kl_start:
            # Compute new weight based on target multiplier and annealing parameters
            new_weight = target_mult * min(2 * (epoch - self.kl_start - (current_cycle - 1) * self.kl_steep) / self.kl_steep, self.anneal_target)

            if (epoch - self.kl_start + 1) % self.kl_steep == 0:
                # Increment cycle counter
                self.cycle += 1

            self.weight = new_weight


    # Loading model function (best model)

    def load_model(self) -> str:  

        """
        Loaded best model from saved file

        Args:
            None: Function contain no arguments

        Returns:
            str: Message indicating that best model has been loaded
        """     

        self.model.load_state_dict(torch.load(os.path.join(self.dirpath, self.best_model_dir, self.filename_best_model)))
        return f"Best model loading ({self.filename_best_model})..."
    
     
    # Prediction function (estimated Hawkes parameters)

    @torch.no_grad()
    def predict(self, val_x: torch.Tensor, dtype: Optional[torch.dtype] = torch.float32, set_name: Optional[str] = "Validation set") -> Tuple[torch.Tensor, float, float]:

        """
        Estimated Hawkes parameters using validation dataset

        Args:
            val_x (torch.Tensor): Input tensor for validation dataset
            dtype (torch.dtype): Type for tensor operations (default: torch.float32)
            set_name (str): Dataset name (default: Validation set)

        Returns:
            Tuple[torch.Tensor, float, float]: Estimations for branching ratio (eta), and baseline intensity (mu)
        """        

        # Evaluation mode
        self.model.eval()

        # Forward pass
        y_pred, mean_pred, log_var_pred = self.model(val_x)

        # Predictions Averages
        val_eta_pred = torch.mean(y_pred[:, -2], dtype=dtype).item()
        val_mu_pred = torch.mean(y_pred[:, -1], dtype=dtype).item()

        print(f"{set_name} - Estimated branching ratio (η): {val_eta_pred:.4f}, Estimated baseline intensity (µ): {val_mu_pred:.4f}")

        return y_pred, mean_pred, log_var_pred


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

        # Started training
        with tqdm(total=self.max_epochs, desc='Training Progress', colour='green') as pbar:
            
            for epoch in range(self.max_epochs):

                # Converged (Fitted) to optimal parameters
                self.train_losses[epoch] = self.run_epoch(train_loader)

                # Evaluated on validation set
                self.val_losses[epoch] = self.evaluate(val_loader)

                # Checked early stopping
                if self.cyclical_annealing(epoch):
                    break

                # Updated progress bar description
                pbar.set_description(f"Epoch {epoch + 1}/{self.max_epochs} - train_loss: {self.train_losses[epoch]:.4f}, val_loss: {self.val_losses[epoch]:.4f}, annealing: {self.weight:.4f}")

                # Updated progress bar
                pbar.update(1)

                # Added losses in TensorBoard at each epoch
                writer.add_scalars("Loss", {"Training": self.train_losses[epoch], "Validation": self.val_losses[epoch]}, epoch)

        # Loaded best model
        print(self.load_model())

        # Computed estimated parameters for validation set (After loaded best model)
        y_pred, _, _ = self.predict(val_x)

        # Added results histograms to TensorBoard
        writer.add_histogram("Intensities Prediction Histogram", y_pred[:, :-2], len(val_y), bins="auto")
        writer.add_histogram("Branching Ratio Prediction Histogram", y_pred[:, -2], len(val_y), bins="auto")
        writer.add_histogram("Baseline Intensity Prediction Histogram", y_pred[:, -1], len(val_y), bins="auto")
        writer.add_histogram("Validation Histogram", val_y, len(val_y), bins="auto")

        # Stored on disk / Closed SummaryWriter
        writer.flush()
        writer.close()
        
        # Written parameters to Parquet file
        write_parquet(pl.DataFrame({'train_losses': self.train_losses, 
                                    'val_losses': self.val_losses}), 
                                    filename=f"{self.run_name}_losses.parquet", 
                                    folder=os.path.join(self.logdirun, self.train_dir, self.run_name))

        write_parquet(pl.DataFrame({'x_true': val_y.numpy(), 
                                    'intensities_pred': y_pred[:, :-2].numpy(),
                                    'eta_pred': y_pred[:, -2].numpy(),
                                    'mu_pred': y_pred[:, -1].numpy()}), 
                                    filename=f"{self.run_name}_predictions.parquet", 
                                    folder=os.path.join(self.logdirun, self.train_dir, self.run_name))

        return self.model, self.train_losses, self.val_losses, y_pred
    

    # Testing fonction (PyTorch Profiler = disable)
    
    @profiling(enable=False)
    def test_model(self, test_x: torch.Tensor, test_y: torch.Tensor) -> torch.Tensor:

        """
        Tested and evaluated model

        Args:
            test_x (torch.Tensor): Features inputs for esting data
            test_y (torch.Tensor): Label outputs for testing data

        Returns:
            torch.Tensor: Intensities predictions
        """

        # Initialized Tensorboard
        writer = SummaryWriter(os.path.join(self.logdirun, self.test_dir, self.run_name))
        
        # Loaded best model
        print(self.load_model())

        # Forward pass for predictions
        y_pred, _, _ = self.predict(test_x, set_name="Test set")

        # Added results histograms to TensorBoard
        writer.add_histogram("Intensities Prediction Histogram", y_pred[:, :-2], len(test_y), bins="auto")
        writer.add_histogram("Branching Ratio Prediction Histogram", y_pred[:, -2], len(test_y), bins="auto")
        writer.add_histogram("Baseline Intensity Prediction Histogram", y_pred[:, -1], len(test_y), bins="auto")
        writer.add_histogram("Test Histogram", test_y, len(test_y), bins="auto")

        # Stored on disk / Closed SummaryWriter
        writer.flush()
        writer.close()

        # Written parameters to parquet file
        write_parquet(pl.DataFrame({'x_true': test_y.numpy(), 
                                    'intensities_pred': y_pred[:, :-2].numpy(),
                                    'eta_pred': y_pred[:, -2].numpy(),
                                    'mu_pred': y_pred[:, -1].numpy()}), 
                                    filename=f"{self.run_name}_predictions.parquet", 
                                    folder=os.path.join(self.logdirun, self.test_dir, self.run_name))

        return y_pred
    

    # Stan testing function

    # def stan_test(self, test_y: torch.Tensor, record: bool = True, filename: Optional[str] = "parameters_predictions.parquet") -> pl.DataFrame:

    #     """
    #     Stan testing for bayesian inference 

    #     Args:
    #         test_y (torch.Tensor): Label outputs for testing data
    #         record (bool, optional): Record results in parquet file (default: True)
    #         filename (str, optional): Parquet filename to save results (default: "parameters_predictions.parquet")

    #     Returns:
    #         pl.DataFrame: Parameters predictions
    #     """

    #     weights = self.model.intensities_decoder.parameters()

    #     W1, B1, W2, B2, W3, B3 = weights

    #     # Initialized testing data
    #     data = {"p": self.latent_size, "p1": int(self.intermediate_size * 0.5), "p2": self.intermediate_size, 
    #             "n": self.input_size, "W1": W1, "B1": B1, "W2": W2, "B2": B2, "W3": W3, "B3": B3, "y": test_y[0,:]}
        
    #     # Open file and read code
    #     with io.open("stan_test.stan", 'r', encoding='utf-8') as file:
    #         program_code = file.read()

    #     # Built stan model
    #     mcmc_fit = stan.build(program_code=program_code, data=data)

    #     # MCMC Adjustment
    #     mcmc_result = mcmc_fit.sample()

    #     # Extracted samples
    #     latent_draws = mcmc_result['z']

    #     # Fitting model from samples
    #     test_pred = self.model.parameters_decoder.fit(latent_draws)

    #      # Written parquet file
    #     if record is True:
    #         write_parquet(pl.DataFrame(np.column_stack((test_pred[:, 0], test_pred[:, 1])), schema=["eta_pred", "mu_pred"]), filename=filename)

    #     return pl.DataFrame(np.column_stack((test_pred[:, 0], test_pred[:, 1])), schema=["eta_pred", "mu_pred"])