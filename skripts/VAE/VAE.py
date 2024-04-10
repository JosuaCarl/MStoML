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

# Argument parser
parser = argparse.ArgumentParser(prog='VAE_smac_run',
                                description='Hyperparameter tuning for Variational Autoencoder with SMAC')
parser.add_argument('-d', '--data_dir')
parser.add_argument('-r', '--run_dir')
parser.add_argument('-v', '--verbosity')
parser.add_argument('-f', '--framework')
args = parser.parse_args()


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["KERAS_BACKEND"] = args.framework
import keras
from keras import Model
from keras import layers
from keras.layers import Input, Dense, Dropout
from keras.losses import MeanSquaredError
from keras import optimizers
from keras import activations
from keras import backend
from keras.callbacks import TensorBoard


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()


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
        if "torch" in framework:
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
    solvers = {"adam": optimizers.legacy.Adam, "nadam": optimizers.legacy.Nadam, "adamw": optimizers.AdamW}
    return solvers[solver]


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

class FIA_VAE():
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
        mse = MeanSquaredError()
        self.reconstruction_loss = mse(true, pred)
        self.kl_loss = backend.mean(-0.5 * backend.sum( 1.0 + self.sigma - backend.square(self.mu) - backend.exp(self.sigma), axis=-1))
        self.loss = self.reconstruction_loss + self.kl_loss

        return self.loss
    
    def train(self, training_data_in, training_data_out, validation_data_in, validation_data_out,
              epochs:int, batch_size:int, callbacks:list, verbosity:int=0):
        self.vae.fit(training_data_in, training_data_out,
                     validation_data = (validation_data_in, validation_data_out),
                     epochs = epochs, batch_size = batch_size, 
                     callbacks = callbacks, verbose = verbosity)
    
    def evaluate(self, test_data_in, test_data_out, verbosity:int=0):
        loss, mse = self.vae.evaluate(test_data_in, test_data_out, verbose=verbosity)
        return (loss, mse)
    

if __name__ == "__main__":
    main()