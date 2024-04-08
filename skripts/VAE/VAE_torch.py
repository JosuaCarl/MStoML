import os
import sys
import time
import random

import numpy as np

from sklearn.model_selection import train_test_split

from dataclasses import dataclass
from ConfigSpace import Configuration, ConfigurationSpace

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchinfo import summary
from GPUtil import showUtilization, getAvailable
import psutil

sys.path.append("..")
from helpers.pc_stats import *

# print(f"Available GPUs: {getAvailable(limit=4)}")

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


# Helpers
def get_activation_function(activation_function:str) -> nn.Module:
    """
    Convert an activation function string into a pytorch function

    Args:
        activation_function (str): Activation function in string representation 

    Returns:
        Activation function as a pytorch.nn class
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

    Args:
        solver (str): Solver in string representation 
    
    Returns:
        solver as a pytorch.optim class
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
    """
    Flow injection analyis - variational autoencoder
    """
    def __init__(self, config:Configuration):
        super(FIA_VAE, self).__init__()
        im_layers       = config["intermediate_layers"]
        im_dim          = config["intermediate_dimension"]
        activation_fun  = get_activation_function( config["intermediate_activation"] )
        
        # Encoder construction
        encoder_layers = [ nn.Dropout(config["input_dropout"]),
                           nn.Linear(config["original_dim"],  im_dim),
                           activation_fun ]
        for i in range(1, im_layers):                                                              # Successive halfing of layers
            if im_dim // 2**i <= config["latent_dimension"]:
                im_layers = i 
                break
            encoder_layers.append( nn.Linear( im_dim // 2**(i-1), im_dim // 2**i ) )
            encoder_layers.append( activation_fun )
        encoder_layers.append( nn.Linear(im_dim // 2**(im_layers-1), 2 * config["latent_dimension"]) )
        self.encoder = nn.Sequential( *encoder_layers)

        # Kernel trick construction
        self.softplus = nn.Softplus()

        # Decoder construction
        decoder_layers = [ nn.Linear(config["latent_dimension"], im_dim // 2**(im_layers-1)),
                           activation_fun ]
        for i in reversed(range(1, im_layers)):
            decoder_layers.append( nn.Linear( im_dim // 2**i, im_dim // 2**(i-1) ) )
            decoder_layers.append( activation_fun )
        decoder_layers.append( nn.Linear(im_dim, config["original_dim"]) )
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
        else:
            loss = None
            kl_loss = None
            reconstruction_loss = None
        
        return VAEOutput(z_dist=dist, z_sample=z, recon_x=recon_x, 
                         loss=loss, reconstruction_loss=reconstruction_loss, kl_loss=kl_loss)
    
    def save_state(self, path:str):
        """
        Save the model parameters

        Args:
            path: Path to the output file
        """
        torch.save(self.state_dict(), path)
    
    def load_state(self, path:str):
        """
        Load the model parameters / state

        Args:
            path: Path to the input file
        """
        self.load_state_dict(torch.load(path))
    


class FIA_VAE_hptune:
    def __init__(self, X, test_size:float, configuration_space:ConfigurationSpace, model_builder,
                 device:str, workers:int=1, batch_size:int=16, verbosity:int=0):
        """
        X: Tensor
        test_size: The fraction as a float to be tested upon
        configuration_space: ConfigurationSpace from configspace (hyperparameters)
        device: device to be used for computation
        batch_size: Number of samples to be used at once as an integer
        verbosity: verbosity level as an integer [0: No output, 1: Summary output, >1: Model and usage output]
        """
        self.device                 = torch.device( device )
        self.configuration_space    = configuration_space
        self.model_builder          = model_builder
        training_data, test_data    = train_test_split(X, test_size=test_size)
        self.training_data          = torch.tensor(training_data.values).to(torch.float32).to( device )
        self.test_data              = torch.tensor(test_data.values).to(torch.float32).to ( device )
        self.workers                = workers
        self.batch_size             = batch_size
        self.verbosity              = verbosity
    
    def seed_worker(self, worker_id):
        """
        Initalizes all modules with relevant random number generators to a seed to control randomness

        Args:
            worker_id: Used to be able to receive input from different workers
        """
        worker_seed = torch.initial_seed()
        random.seed(worker_seed)
        rng = np.random.default_rng(worker_seed)

    def train_epoch(self, model, data_loader, optimizer):
        """
        Train the model for one epoch

        Args:
            model
            data_loader
            optimizer
        """
        for data in data_loader:
            optimizer.zero_grad()  # Zero the gradients
            output = model(data)  # Forward pass
            loss = output.loss
            loss.backward()
            optimizer.step()  # Update the model parameters
    
    def evaluate(self, model, data_loader):
        """
        Evaluate the model

        Args:
            model
            data_loader
        
        Returns:
            Average loss of the model
        """
        sum_loss = 0
        for data in data_loader:
            with torch.no_grad():
                output = model(data) 
                sum_loss += output.loss
        return sum_loss / len(data_loader)
    
    def train(self, config: Configuration, seed:int=0, budget:int=10):
        """
        Method to train the model

        Args:
            config: Configuration to be trained upon
            seed: initializing seed
            budget: number of epochs to be used in training
        
        Returns:
            Average loss of the model
        """
        t = time.time()        
        generator = torch.Generator()
        generator.manual_seed(seed)

        # Dataset Loading
        train_loader = DataLoader(self.training_data, batch_size=self.batch_size,
                                  num_workers=self.workers, worker_init_fn=self.seed_worker,
                                  generator=generator, pin_memory=False )
        test_loader = DataLoader(self.test_data, batch_size=self.batch_size,
                                  num_workers=self.workers, worker_init_fn=self.seed_worker,
                                  generator=generator, pin_memory=False
                                )
        
        # Definition
        model = self.model_builder(config).to( self.device )
        if self.verbosity > 1:
            if self.verbosity > 2:
                print(model)
                summary(model, inpute_size=self.training_data.shape, mode="train", device=self.device)
                print_utilization()
            print(f"Model built in {time.time()-t}s")
            t = time.time()
        
        # Fitting
        optimizer = get_solver( config["solver"] )(model.parameters(), lr=config["learning_rate"]) 
        for epoch in range(int(budget)):
            self.train_epoch(model=model, data_loader=train_loader, optimizer=optimizer)
        if self.verbosity > 1:
            if self.verbosity > 2:
                print("After training utilization:")
                print_utilization()
            print(f"Model trained in {time.time()-t}s")
            t = time.time()
        
        # Evaluation
        avg_loss = self.evaluate(model, test_loader)
        if self.verbosity > 1:
            print(f"Model evaluated in {time.time()-t}s")

        return avg_loss.cpu()