import sys
import gc
import os
import time
import random

from typing import Union
from tqdm import tqdm
from pathlib import Path
from functools import partial

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from ConfigSpace import Categorical, Configuration, ConfigurationSpace, EqualsCondition, Float, InCondition, Integer, Constant, ForbiddenGreaterThanRelation
from smac import MultiFidelityFacade, HyperparameterOptimizationFacade
from smac import Scenario
from smac.intensifier.hyperband import Hyperband
from smac.runhistory.dataclasses import TrialValue


sys.path.append( '..' )
from helpers import *

# Helpers
def get_activation_function(activation_function:str):
    """
    Convert an activation function string into a pytorch function
    """
    if activation_function == "relu":
        return nn.ReLU()
    elif activation_function == "leakyrelu":
        return nn.LeakyReLU()
    elif activation_function == "selu":
        return nn.SELU()
    else:
        raise(ValueError(f"{activation_function} is not defined as a valid activation function."))
    
def get_solver(solver:str):
    """
    Convert an solver string into a pytorch function
    """
    if solver == "adam":
        return optim.Adam
    elif solver == "nadam":
        return optim.NAdam
    elif solver == "adamw":
        return optim.AdamW
    else:
        raise(ValueError(f"{solver} is not defined as a valid activation function."))


# VAE
class FIA_VAE(nn.Module):
    def __init__(self, config:Configuration):
        super(FIA_VAE, self).__init__()

        self.encoder = nn.Sequential( 
            nn.Dropout(config["input_dropout"]),
            nn.Linear(config["original_dim"],  config["intermediate_dimension"])
        )
        for i in range(1, config["intermediate_layers"]):
            self.encoder.add( nn.Linear(config["intermediate_dimension"] // 2**(i-1),  config["intermediate_dimension"] // 2**i ) )
            self.encoder.add( get_activation_function(config["intermediate_activation"]) )
        self.encoder.add( nn.Linear(config["intermediate_dimension"] // 2**config["Layers"], 2 * config["latent_dimension"]) )

        self.softplus = nn.Softplus()

        self.decoder = nn.Sequential(
            nn.Linear(config["latent_dimension"], config["intermediate_dimension"] // 2**config["Layers"]),
            )
        for i in range(config["intermediate_layers"]-1, -1, -1):
            self.decoder.add( nn.Linear(config["intermediate_dimension"] // 2** i,  config["intermediate_dimension"] // 2**(i-1) ) )
            self.decoder.add( get_activation_function(config["intermediate_activation"]) )
        self.decoder.add( nn.Sigmoid() )

    def encode(self, x, eps: float = 1e-8):
        """
        Encodes the input data into the latent space.
        
        Args:
            x (torch.Tensor): Input data.
            eps (float): Small value to avoid numerical instability.
        
        Returns:
            torch.distributions.MultivariateNormal: Normal distribution of the encoded data.
        """
        x = self.encoder(x)
        mu, logvar = torch.chunk(x, 2, dim=-1)
        scale = self.softplus(logvar) + eps
        scale_tril = torch.diag_embed(scale)
        
        return torch.distributions.MultivariateNormal(mu, scale_tril=scale_tril)
        
    def reparameterize(self, dist):
        """
        Reparameterizes the encoded data to sample from the latent space.
        
        Args:
            dist (torch.distributions.MultivariateNormal): Normal distribution of the encoded data.
        Returns:
            torch.Tensor: Sampled data from the latent space.
        """
        return dist.rsample()
    
    def decode(self, z):
        """
        Decodes the data from the latent space to the original input space.
        
        Args:
            z (torch.Tensor): Data in the latent space.
        
        Returns:
            torch.Tensor: Reconstructed data in the original input space.
        """
        return self.decoder(z)
    
    def forward(self, x, compute_loss: bool = True):
        """
        Performs a forward pass of the VAE.
        
        Args:
            x (torch.Tensor): Input data.
            compute_loss (bool): Whether to compute the loss or not.
        
        Returns:
            VAEOutput: VAE output dataclass.
        """
        dist = self.encode(x)
        z = self.reparameterize(dist)
        recon_x = self.decode(z)
        
        if compute_loss:
            # compute loss terms 
            reconstruction_loss = F.MSE(recon_x, x + 0.5, reduction='none').sum(-1).mean()
            std_normal = torch.distributions.MultivariateNormal(
                torch.zeros_like(z, device=z.device),
                scale_tril=torch.eye(z.shape[-1], device=z.device).unsqueeze(0).expand(z.shape[0], -1, -1),
            )
            kl_loss = torch.distributions.kl.kl_divergence(dist, std_normal).mean()
                    
            loss = reconstruction_loss + self.config["kl_loss_scaler"] * kl_loss
        
        return {"z_dist":dist, "z_sample":z, "recon_x":recon_x, 
                "loss":loss, "reconstruction_loss":reconstruction_loss, "kl_loss":kl_loss}
    


class FIA_VAE_hptune:
    def __init__(self, X, test_size:float, configuration_space:ConfigurationSpace, model_builder, device):
        self.configuration_space = configuration_space
        self.model_builder = model_builder
        training_data, test_data = train_test_split(X, test_size=test_size)
        self.training_data = torch.tensor(training_data.values).to(device)
        self.test_data = torch.tensor(test_data.values).to(device)
        self.device = device
    """
    def train(self, model, data, optimizer, epochs):
        for epoch in range(epochs):
            optimizer.zero_grad()  # Zero the gradients
            output = model(data)  # Forward pass
            loss = output.loss
            loss.backward()
            optimizer.step()  # Update the model parameters
    """
    def evaluate(self, model, data):
        output = model(data) 
        return output.loss
    
    def train(self, config: Configuration, seed:int=0, budget:int=10):        
        torch.manual_seed(seed)
        random.seed(seed)

        model = self.model_builder(config).to( self.device )
        model.train()  # Set the model to training mode

        # Training
        optimizer = get_solver( config["solver"] )(model.parameters(), lr=config["learning_rate"])        
        for epoch in range(budget):
            optimizer.zero_grad()  # Zero the gradients
            output = model(self.training_data)  # Forward pass
            loss = output.loss
            loss.backward()
            optimizer.step()  # Update the model parameters

        # Evaluation
        loss = self.evaluate(model, self.test_data)

        return loss