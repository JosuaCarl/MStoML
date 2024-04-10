import sys
import os
import argparse
from tqdm import tqdm
from pathlib import Path
import time
import datetime
import random
import numpy as np
import pandas as pd

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
from torch.utils.tensorboard import SummaryWriter

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import keras
from keras import Model
from keras import layers
from keras.layers import Input, Dense, Dropout
from keras.losses import mse
from keras import optimizers
from keras import activations
from keras import backend
from keras.callbacks import TensorBoard
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

from dataclasses import dataclass
from ConfigSpace import Configuration, ConfigurationSpace

sys.path.append( '..' )
from FIA.FIA import total_ion_count_normalization
from helpers.pc_stats import *

# Argument parser
parser = argparse.ArgumentParser(prog='VAE_smac_run',
                                description='Hyperparameter tuning for Variational Autoencoder with SMAC')
parser.add_argument('-d', '--data_dir')
parser.add_argument('-r', '--run_dir')
parser.add_argument('-v', '--verbosity')
parser.add_argument('-f', '--framework')
args = parser.parse_args()


def main():
    """
    Hyperparameter optimization with SMAC3
    """
    data_dir, run_dir = [os.path.normpath(os.path.join(os.getcwd(), d)) for d in  [args.data_dir, args.run_dir]]
    verbosity =  int(args.verbosity)
    framework = args.framework
    outdir = Path(os.path.normpath(os.path.join(run_dir, f"smac_vae_{framework}")))
    if verbosity > 0:
        print_available_gpus()
        if "keras" in framework:
            print("Available GPUs: ", tf.config.list_physical_devices('GPU'))
            
    time_step(message="Setup loaded", verbosity=verbosity)

    X = read_data(data_dir, verbosity=verbosity)

    config_space = ConfigurationSpace(
                {'input_dropout': 0.1, 'intermediate_activation': "relu", 'intermediate_dimension': 500,
                'intermediate_layers': 4, 'latent_dimension': 10, 'learning_rate': 0.001,
                'original_dim': 825000, 'solver': 'nadam'}
            )
    config = config_space.get_default_configuration()
    seed = 42

    if "torch" in framework:
        device = search_device(verbosity=verbosity)
        model = FIA_VAE_torch( config )
        
    elif "keras" in framework:
        model = FIA_VAE_keras( config )

    

    generator = torch.Generator()
    generator.manual_seed(seed)
    data = torch.tensor(X.iloc[0].values).to( torch.float32 ).to( device )
    data_loader = DataLoader(data, batch_size=len(data), num_workers=0, worker_init_fn=seed_worker, generator=generator, pin_memory=False )

    model.init_weights()
    optimizer = get_solver( config["solver"] )(model.parameters(), lr=config["learning_rate"])
    for epoch in tqdm(range(10)):
        for data in data_loader:
            optimizer.zero_grad()  # Zero the gradients
            output = model(data)  # Forward pass
            loss = output.loss
            loss.backward()
            optimizer.step()  # Update the model parameters
    print(f"Final Loss: {loss}")

    model.train(config.get_default_configuration(), seed=42, budget=2)


def read_data(data_dir:str, verbosity:int=0):
    """
    Read in the data from a data_matrix and normalize it according to total ion counts

    Args:
        data_dir (str): Directory with "data_matrix.tsv" file. The rows must represent m/z bins, the columns different samples
    Returns:
        X: matrix with total ion count (TIC) normalized data (transposed)
    """
    binned_dfs = pd.read_csv(os.path.join(data_dir, "data_matrix.tsv"), sep="\t", index_col="mz", engine="pyarrow")
    binned_dfs[:] =  total_ion_count_normalization(binned_dfs)

    X = binned_dfs.transpose()
    time_step(message="Data loaded", verbosity=verbosity)
    return X


def time_step(message:str, verbosity:int=0):
    """
    Saves the time difference between last and current step
    """
    global last_timestamp
    global step
    global runtimes
    runtimes[f"{step}: {message}"] = [time.time() - last_timestamp]
    last_timestamp = time.time()
    step += 1
    if verbosity > 0: 
        print(message)


# Helpers
def get_activation_function(framework:str, activation_function:str) -> nn.Module:
    """
    Convert an activation function string into a pytorch function

    Args:
        framework (str): Framework of model
        activation_function (str): Activation function in string representation
    Returns:
        Activation function as a pytorch.nn class
    """
    act_funcs_torch = {"relu": nn.ReLU(), "leakyrelu" :nn.LeakyReLU(), "selu": nn.SELU(), "tanh": nn.Tanh()}
    act_funcs_keras = {"relu": activations.relu, "leakyrelu" :layers.LeakyReLU(), "selu": activations.selu, "tanh": activations.tanh}
    if "torch" in framework:
        return act_funcs_torch[activation_function]
    elif "keras" in framework:
        return act_funcs_keras[activation_function]
        
def get_solver(framework:str, solver:str):
    """
    Convert an solver string into a pytorch function

    Args:
        solver (str): Solver in string representation 
    
    Returns:
        solver as a pytorch.optim class
    """
    solvers_torch = {"adam": optimizers.legacy.Adam, "nadam": optimizers.legacy.Nadam, "adamw": optimizers.AdamW}
    solvers_keras = {"adam": optim.Adam, "nadam": optim.NAdam, "adamw": optim.AdamW}
    if "torch" in framework:
        return solvers_torch[solver]
    elif "keras" in framework:
        return solvers_keras[solver]


def search_device(verbosity:int=0):
    """
    Searches the fastest device for computation in pytorch
    Args:
        verbosity (int): level of verbosity
    Returns:
        device (str)
    """
    device = ( "cuda" if torch.cuda.is_available()
                    else "mps" if torch.backends.mps.is_available()
                    else "cpu" )
    if verbosity > 0:
        print(f"Using {device} device")
    return device

def seed_worker(worker_id):
    """
    Initalizes all modules with relevant random number generators to a seed to control randomness

    Args:
        worker_id: Used to be able to receive input from different workers
    """
    worker_seed = torch.initial_seed()
    random.seed(worker_seed)
    rng = np.random.default_rng(worker_seed)
    
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
class FIA_VAE_torch(nn.Module):
    """
    Flow injection analyis - variational autoencoder
    """
    def __init__(self, config:Configuration):
        super(FIA_VAE_torch, self).__init__()
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

        # Decoder construction
        decoder_layers = [ nn.Linear(config["latent_dimension"], im_dim // 2**(im_layers-1)),
                           activation_fun ]
        for i in reversed(range(1, im_layers)):
            decoder_layers.append( nn.Linear( im_dim // 2**i, im_dim // 2**(i-1) ) )
            decoder_layers.append( activation_fun )
        decoder_layers.append( nn.Linear(im_dim, config["original_dim"]) )
        decoder_layers.append( nn.Sigmoid() )
        self.decoder = nn.Sequential( *decoder_layers)

        # Kernel trick construction
        self.softplus = nn.Softplus()

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

    def init_weights(self):
        def init_kaiming_uniform(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.kaiming_uniform_(module.weight)
        self.encoder.apply(init_kaiming_uniform)
        self.decoder.apply(init_kaiming_uniform)

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



class Sampling(layers.Layer):
    """
    Uses (z_mean, z_log_var) to sample z, the vector encoding a digit.
    """
    def call(self, inputs):
        z_mean, z_log_var = inputs
        z_mean_shape = backend.shape(z_mean)
        batch   = z_mean_shape[0]
        dim     = z_mean_shape[1]
        epsilon = backend.random_normal(shape=(batch, dim))
        return z_mean + backend.exp(0.5 * z_log_var) * epsilon


class FIA_VAE_keras():
    def __init__(self, config:Configuration):
        im_layers       = config["intermediate_layers"]
        im_dim          = config["intermediate_dimension"]
        activation_fun  = get_activation_function( config["intermediate_activation"] )
        
        # Encoder
        self.input      = Input(shape=(config["original_dim"],), name='encoder_input')
        self.im_enc     = Dropout( config["input_dropout"] ) (self.input)
        self.im_enc     = Dense( im_dim , activation=activation_fun ) (self.im_enc)
        for i in range(1, im_layers):                                                              # Successive halfing of layers
            if im_dim // 2**i <= config["latent_dimension"]:
                im_layers = i 
                break
            self.im_enc = Dense( im_dim // 2**i, activation=activation_fun ) (self.im_enc)
        self.mu         = Dense( config["latent_dimension"], name='latent_mu' ) (self.im_enc)
        self.sigma      = Dense( config["latent_dimension"], name='latent_sigma' ) (self.im_enc)
        self.z          = Sampling() ( [self.mu, self.sigma] )                                     # Use reparameterization trick

        self.encoder = Model( self.input, [self.mu, self.sigma, self.z], name='encoder' )            # Instantiate encoder

        # Decoder
        self.decoder_input  = Input(shape=(config["latent_dimension"], ), name='decoder_input')
        prev_layer = self.decoder_input
        for i in reversed(range(1, im_layers)):
            self.im_dec =  Dense( im_dim // 2**i, activation=activation_fun) (prev_layer)
            prev_layer = self.im_dec
        self.im_dec = Dense(im_dim, activation=activation_fun) (prev_layer)
        self.output  = Dense(config["original_dim"]) (self.im_dec)

        self.decoder = Model(self.decoder_input, self.output, name='decoder')                        # Instantiate decoder

        # VAE
        self.vae_outputs = self.decoder(self.encoder(self.input)[2])
        self.vae         = Model(self.input, self.vae_outputs, name='vae')

        # Loss trackers
        self.reconstruction_loss = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss = keras.metrics.Mean(name="kl_loss")
        self.loss = keras.metrics.Mean(name="total_loss")

        # Define optimizer
        self.optimizer = get_solver( config["solver"] )( config["learning_rate"] )

        # Compile VAE
        self.vae.compile(optimizer=self.optimizer, loss=self.kl_reconstruction_loss, metrics = [ "mse" ])

    def encode(self, data):
        return self.encoder.predict(data)[2]
    
    def encode_mu(self, data):
        return self.encoder.predict(data)[0]
    
    def decode(self, data):
        return self.decoder.predict(data)
    
    def reconstruct(self, data):
        return self.decode(self.encode(data))
    
    def save_model(self, save_folder, suffix:str=""):
        self.vae.save(os.path.join(save_folder, f'VAE{suffix}.h5'))
        self.encoder.save(os.path.join(save_folder, f'VAE_encoder{suffix}.h5'))
        self.decoder.save(os.path.join(save_folder, f'VAE_decoder{suffix}.h5'))
        
    def load_vae(self, save_path):                       
        self.vae = keras.models.load_model(save_path)
        self.vae.compile(optimizer=self.optimizer, 
                         loss=self.kl_reconstruction_loss, 
                         metrics = ['mse'])
        
    def load_encoder(self, save_path):
        self.encoder = keras.models.load_model(save_path)
        
    def load_decoder(self, save_path):
        self.decoder = keras.models.load_model(save_path)
    
    def kl_reconstruction_loss(self, true, pred):
        """
        Loss function for Kullback-Leibler + Reconstruction loss

        Args:
            true: True values
            pred: Predicted values
        Returns:
            Loss = Kullback-Leibler + Reconstruction loss
        """
        self.reconstruction_loss = mse(true, pred)
        self.kl_loss = backend.mean(-0.5 * backend.sum( 1.0 + self.sigma - backend.square(self.mu) - backend.exp(self.sigma), axis=-1))
        self.loss = self.reconstruction_loss + self.kl_loss

        return self.loss
    
    def train(self, train_data, val_data, epochs:int, batch_size:int, log_dir:str, verbosity:int=0):
        if verbosity > 0:
            log_dir = os.path.join(log_dir,  datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True,
                                               update_freq='epoch', profile_batch=2, embeddings_freq=1)

            self.vae.fit(train_data, train_data,
                        epochs = epochs, 
                        batch_size = batch_size, 
                        validation_data = (val_data, val_data),
                        verbose = verbosity,
                        callbacks=[ tensorboard_callback ])
    
    def evaluate(self, test_data, verbosity:int=0):
        loss, mse = self.vae.evaluate(test_data, test_data, verbose=verbosity)
        return (loss, mse)



if __name__ == "__main__":
    main()