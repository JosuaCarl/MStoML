import sys
import os
import argparse
from tqdm import tqdm
from pathlib import Path
import time
import datetime
import numpy as np
import pandas as pd

from ConfigSpace import Configuration, ConfigurationSpace

sys.path.append( '..' )
from helpers.normalization import *
from helpers.pc_stats import *


# os.environ["KERAS_BACKEND"] = "torch"             # Set to change backend of keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import keras
from keras import Model, Sequential
from keras.layers import Input, Dense, Dropout
from keras import backend, ops, layers, activations, metrics, losses, optimizers
from keras.callbacks import TensorBoard


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

import tensorflow as tf


# Logging (time and steps)
last_timestamp = time.time()
step = 0
runtimes = {}

def main():
    """
    Hyperparameter optimization with SMAC3
    """
    data_dir, run_dir = [os.path.normpath(os.path.join(os.getcwd(), d)) for d in  [args.data_dir, args.run_dir]]
    verbosity =  int(args.verbosity)
    framework = args.framework
    outdir = Path(os.path.normpath(os.path.join(run_dir, f"smac_vae_{framework}")))
    if verbosity > 0 and "gpu" in framework:
        print_available_gpus()
        if "tensorflow" in framework:
            print("Available GPUs: ", tf.config.list_physical_devices('GPU'))
            
    time_step(message="Setup loaded", verbosity=verbosity, min_verbosity=1)

    X = read_data(data_dir, verbosity=verbosity)

    config_space = ConfigurationSpace(
                {'input_dropout': 0.1, 'intermediate_activation': "relu", 'intermediate_dimension': 500,
                'intermediate_layers': 4, 'latent_dimension': 10, 'learning_rate': 0.001,
                'original_dim': 825000, 'solver': 'nadam'}
            )
    config = config_space.get_default_configuration()
    seed = 42
    epochs = 10000


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
    time_step(message="Data loaded", verbosity=verbosity, min_verbosity=1)
    return X


def time_step(message:str, verbosity:int=0, min_verbosity:int=1):
    """
    Saves the time difference between last and current step
    """
    global last_timestamp
    global step
    global runtimes
    runtimes[f"{step}: {message}"] = time.time() - last_timestamp
    if verbosity >= min_verbosity: 
        print(f"{message} ({runtimes[f'{step}: {message}']}s)")
    last_timestamp = time.time()
    step += 1


def get_activation_function(activation_function:str):
        """
        Convert an activation function string into a keras function

        Args:
            activation_function (str): Activation function in string representation
        Returns:
            Activation function as keras.activations or keras.layers
        """
        activation_functions = {"relu": activations.relu, "leakyrelu" :layers.LeakyReLU(), "selu": activations.selu, "tanh": activations.tanh}
        return activation_functions[activation_function]
    
def get_solver(solver:str):
    """
    Convert an solver string into a keras function

    Args:
        solver (str): Solver in string representation 
    Returns:
        solver as a keras.optimizers class
    """
    solvers = {"adam": optimizers.Adam, "nadam": optimizers.Nadam, "adamw": optimizers.AdamW}
    return solvers[solver]


class Sampling(layers.Layer):
        """
        Uses (z_mean, z_log_var) to sample z, the vector encoding a digit.
        """
        def call(self, inputs):
            z_mean, z_log_var = inputs
            z_mean_shape = ops.shape(z_mean)
            batch   = z_mean_shape[0]
            dim     = z_mean_shape[1]
            epsilon = keras.random.normal(shape=(batch,dim))
            return ops.multiply(ops.add(z_mean, ops.exp(0.5 * z_log_var)), epsilon)

class FIA_VAE(Model):
    def __init__(self, config:Configuration):
        super().__init__()
        intermediate_dims       = [i for i in range(config["intermediate_layers"]) 
                                    if config["intermediate_dimension"] // 2**i > config["latent_dimension"]]
        activation_function     = get_activation_function( config["intermediate_activation"] )

        # Encoder (with sucessive halfing of intermediate dimension)
        self.dropout            = Dropout( config["input_dropout"] , name="dropout")        
        self.intermediate_enc   = Sequential ( [ Input(shape=(config["original_dim"],), name='encoder_input') ] +
                                               [ Dense( config["intermediate_dimension"] // 2**i,
                                                        activation=activation_function ) 
                                                for i in intermediate_dims] , name="encoder_intermediate")

        self.mu_encoder         = Dense( config["latent_dimension"], name='latent_mu' )
        self.sigma_encoder      = Dense( config["latent_dimension"], name='latent_sigma' )
        self.z_encoder          = Sampling(name="latent_reparametrization") 

        # Decoder
        self.decoder            = Sequential( [ Input(shape=(config["latent_dimension"], ), name='decoder_input') ] +
                                              [ Dense( config["intermediate_dimension"] // 2**i,
                                                       activation=activation_function )
                                               for i in reversed(intermediate_dims) ] +
                                              [ Dense(config["original_dim"]) ] , name="Decoder")

        # Loss trackers
        self.reconstruction_loss    = metrics.Mean(name="reconstruction_loss")
        self.kl_loss                = metrics.Mean(name="kl_loss")
        self.loss_tracker           = metrics.Mean(name="loss")

        # Define optimizer
        self.optimizer = get_solver( config["solver"] )( config["learning_rate"] )

        # Compile VAE
        self.compile(optimizer=self.optimizer, loss=self.kl_reconstruction_loss, metrics = [ "mse" ])

    @property
    def metrics(self):
        return [self.loss_tracker, self.reconstruction_loss, self.kl_loss]

    def call(self, data, training=False):
        x = self.dropout(data, training=training)
        return self.decode(self.encode(x))

    def encode(self, data):
        x = self.intermediate_enc(data)
        self.mu = self.mu_encoder(x)
        self.sigma = self.sigma_encoder(x)
        self.z = self.z_encoder( [self.mu, self.sigma] )
        return self.z
    
    def encode_mu(self, data):
        x = self.intermediate_enc(data)
        return self.mu(x)
    
    def decode(self, x):
        return self.decoder(x)
    
    def save_model(self, save_folder, suffix:str=""):
        self.save(os.path.join(save_folder, f'VAE{suffix}.h5'))
         
    def kl_reconstruction_loss(self, y_true, y_pred):
        """
        Loss function for Kullback-Leibler + Reconstruction loss

        Args:
            true: True values
            pred: Predicted values
        Returns:
            Loss = Kullback-Leibler + Reconstruction loss
        """
        reconstruction_loss = losses.mean_absolute_error(y_true, y_pred)
        self.reconstruction_loss.update_state(reconstruction_loss)
        kl_loss = -0.5 * ops.sum( 1.0 + self.sigma - ops.square(self.mu) - ops.exp(self.sigma) )
        self.kl_loss.update_state(kl_loss)
        loss = reconstruction_loss + kl_loss
        self.loss_tracker.update_state( loss )
        return loss
    
    """
    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute our own loss
            loss = self.kl_reconstruction_loss(y, y_pred)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        return {"loss": self.loss_tracker.result(),
                "reconstruction_loss": self.reconstruction_loss.result(), 
                "kl_loss": self.kl_loss.result()}
    """


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(prog='VAE_smac_run',
                                     description='Hyperparameter tuning for Variational Autoencoder with SMAC')
    parser.add_argument('-d', '--data_dir')
    parser.add_argument('-r', '--run_dir')
    parser.add_argument('-v', '--verbosity')
    parser.add_argument('-f', '--framework')
    args = parser.parse_args()
 
    main()