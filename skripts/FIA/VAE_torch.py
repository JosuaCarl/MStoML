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

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

from torchinfo import summary

from ConfigSpace import Categorical, Configuration, ConfigurationSpace, EqualsCondition, Float, InCondition, Integer, Constant, ForbiddenGreaterThanRelation
from smac import MultiFidelityFacade, HyperparameterOptimizationFacade
from smac import Scenario
from smac.intensifier.hyperband import Hyperband
from smac.runhistory.dataclasses import TrialValue


sys.path.append( '..' )
from helpers import *



# Helpers
def get_activation_function(activation_function:str) -> nn.Module:
    """
    Convert an activation function string into a pytorch function
    """
    if activation_function == "relu":
        return nn.ReLU()
    elif activation_function == "leakyrelu":
        return nn.LeakyReLU()
    elif activation_function == "selu":
        return nn.SELU()
    elif activation_function == "tanh":
        return nn.Tanh()
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
        raise(ValueError(f"{solver} is not defined as a valid solver."))
    
# Dataclass
@dataclass
class VAEOutput:
    """
    Dataclass for VAE output.
    
    Attributes:
        z_dist (torch.distributions.Distribution): The distribution of the latent variable z.
        z_sample (torch.Tensor): The sampled value of the latent variable z.
        x_recon (torch.Tensor): The reconstructed output from the VAE.
        loss (torch.Tensor): The overall loss of the VAE.
        loss_recon (torch.Tensor): The reconstruction loss component of the VAE loss.
        loss_kl (torch.Tensor): The KL divergence component of the VAE loss.
    """
    z_dist: torch.distributions.Distribution
    z_sample: torch.Tensor
    recon_x: torch.Tensor
    
    loss: torch.Tensor
    reconstruction_loss: torch.Tensor
    kl_loss: torch.Tensor


# VAE
class FIA_VAE(nn.Module):
    def __init__(self, config:Configuration):
        super(FIA_VAE, self).__init__()
        layers = config["intermediate_layers"]
        
        # Encoder construction
        encoder_layers = [ nn.Linear(config["original_dim"],  config["intermediate_dimension"]),
                           nn.Dropout(config["input_dropout"]), 
                           get_activation_function( config["intermediate_activation"] ) ]
        for i in range(1, layers):
            layers = i
            if config["intermediate_dimension"] // 2**i <= config["latent_dimension"]:
                break
            encoder_layers.append( nn.Linear( config["intermediate_dimension"] // 2**(i-1), config["intermediate_dimension"] // 2**i ) )
            encoder_layers.append( get_activation_function( config["intermediate_activation"] ) )
        encoder_layers.append( nn.Linear(config["intermediate_dimension"] // 2**(layers-1), 2 * config["latent_dimension"]) )
        self.encoder = nn.Sequential( *encoder_layers)

        # Kernel trick construction
        self.softplus = nn.Softplus()

        # Decoder construction
        decoder_layers = [ nn.Linear(config["latent_dimension"], config["intermediate_dimension"] // 2**(layers-1)),
                           get_activation_function( config["intermediate_activation"] ) ]
        for i in reversed(range(1, layers)):
            decoder_layers.append( nn.Linear( config["intermediate_dimension"] // 2**i, config["intermediate_dimension"] // 2**(i-1) ) )
            decoder_layers.append( get_activation_function( config["intermediate_activation"] ) )
        decoder_layers.append( nn.Linear(config["intermediate_dimension"], config["original_dim"]) )
        decoder_layers.append( nn.Sigmoid() )
        self.decoder = nn.Sequential( *decoder_layers)


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
            reconstruction_loss = F.mse_loss(recon_x, x, reduction='mean')
            std_normal = torch.distributions.MultivariateNormal(
                torch.zeros_like(z, device=z.device),
                scale_tril=torch.eye(z.shape[-1], device=z.device).unsqueeze(0).expand(z.shape[0], -1, -1),
            )
            kl_loss = torch.distributions.kl.kl_divergence(dist, std_normal).mean()
                    
            loss = reconstruction_loss + kl_loss
        
        return VAEOutput(z_dist=dist, z_sample=z, recon_x=recon_x, 
                         loss=loss, reconstruction_loss=reconstruction_loss, kl_loss=kl_loss)
    


class FIA_VAE_hptune:
    def __init__(self, X, test_size:float, configuration_space:ConfigurationSpace, model_builder,
                 device:str, workers:int=1, batch_size:int=16, verbose:bool=False):
        self.configuration_space    = configuration_space
        self.model_builder          = model_builder
        training_data, test_data    = train_test_split(X, test_size=test_size)
        self.training_data          = torch.tensor(training_data.values).to(torch.float32).to(device)
        self.test_data              = torch.tensor(test_data.values).to(torch.float32).to(device)
        self.device                 = device
        self.workers                = workers
        self.batch_size             = batch_size
        self.verbose                = verbose
    
    def seed_worker(self, worker_id):
        worker_seed = torch.initial_seed()
        random.seed(worker_seed)
        rng = np.random.default_rng(worker_seed)

    def train_epoch(self, model, data_loader, optimizer):
        for data in data_loader:
            optimizer.zero_grad()  # Zero the gradients
            output = model(data)  # Forward pass
            loss = output.loss
            loss.backward()
            optimizer.step()  # Update the model parameters
    
    def evaluate(self, model, data_loader):
        sum_loss = 0
        for data in data_loader:
            with torch.no_grad():
                output = model(data) 
                sum_loss += output.loss
        return sum_loss / len(data_loader)
    
    def train(self, config: Configuration, seed:int=0, budget:int=10):
        t = time.time()        
        generator = torch.Generator()
        generator.manual_seed(seed)

        train_loader = DataLoader(self.training_data, batch_size=self.batch_size,
                                  num_workers=self.workers, worker_init_fn=self.seed_worker,
                                  generator=generator, pin_memory=True
                                  )

        model = self.model_builder(config).to( self.device )
        if self.verbose:
            print(str(summary(model, inpute_size=self.training_data.shape, mode="train", device=self.device)))
        model.train(True)  # Set the model to training mode
        if self.verbose:
            print(f"Model built in {time.time()-t}s")
            t = time.time()
        
        optimizer = get_solver( config["solver"] )(model.parameters(), lr=config["learning_rate"]) 
        for epoch in range(int(budget)):
            self.train_epoch(model=model, data_loader=train_loader, optimizer=optimizer)
        if self.verbose:
            print(f"Model trained in {time.time()-t}s")
            t = time.time()

        test_loader = DataLoader(self.test_data, batch_size=self.batch_size,
                                  num_workers=self.workers, worker_init_fn=self.seed_worker,
                                  generator=generator, pin_memory=True
                                )
        avg_loss = self.evaluate(model, test_loader)
        if self.verbose:
            print(f"Model evaluated in {time.time()-t}s")

        return avg_loss