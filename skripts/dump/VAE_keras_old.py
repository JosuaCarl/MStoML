import sys
import gc
import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

from typing import Union
from tqdm import tqdm
from pathlib import Path
from functools import partial

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import tensorflow as tf
from keras import Model
from keras.layers import Input, Dense, LeakyReLU, Lambda, Dropout, BatchNormalization
from keras.losses import mse
from keras.optimizers.legacy import Nadam
import keras.backend as backend
import keras

from ConfigSpace import Configuration, ConfigurationSpace

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()


sys.path.append( '..' )
from skripts.helpers.helpers import *

print("Available GPUs: ", tf.config.list_physical_devices('GPU'))

# Methods
def sample_z(args):
    """
    Define sampling with reparameterization trick
    """
    mu, sigma = args
    batch     = backend.shape(mu)[0]
    dim       = backend.shape(mu)[1]
    eps       = backend.random_normal(shape=(batch, dim))
    return mu + backend.exp(sigma / 2) * eps


def kl_reconstruction_loss(true, pred, sample_weight, mu, sigma, original_dim):
        """
        Define loss function for Kullback-Leibler and Reconstruction loss

        Args:
            true: True values
            pred: Predicted valeus
            sample_weight: Weights of each sample
            mu: Mean of the underlying distribution
            sigma: Standard deviation of the underlying distribution
            original_dim: Original number of m/z bins
        Returns:
            Mean of the reconstruction and Kullback-Leibler loss across all dimensions
        """
        reconstruction_loss = mse(true, pred) * original_dim
        kl_loss = -0.5 * backend.sum( 1.0 + sigma - backend.square(mu) - backend.exp(sigma), axis=-1)

        return backend.mean(reconstruction_loss + kl_loss)


# Tuning VAE
def build_vae_ht_model(config:Configuration):
    t = time.time()
    backend.clear_session()
    gc.collect()
    print(f"Garbage collected ({time.time()-t}s)")
    t = time.time()
    intermediate_activation = config["intermediate_activation"]
    if intermediate_activation == "leakyrelu":
        intermediate_activation = LeakyReLU()

    # Encoder
    i       = Input(shape=(config["original_dim"], ), name='encoder_input')
    enc     = Dropout( config["input_dropout"] ) (i)      # Dropout for more redundant neurons
    enc     = Dense( config["intermediate_neurons"], activation=intermediate_activation ) (i)
    mu      = Dense( config["latent_dimensions"], name='latent_mu') (enc)
    sigma   = Dense( config["latent_dimensions"], name='latent_sigma') (enc)
    z       = Lambda(sample_z, output_shape=( config["latent_dimensions"], ), name='z') ([mu, sigma])  ## Use reparameterization trick

    encoder = Model(i, [mu, sigma, z], name='encoder')  ## Instantiate encoder
    
    # Decoder
    d_i     = Input(shape=( config["latent_dimensions"], ), name='decoder_input')
    dec     = Dense( config["intermediate_neurons"], activation=intermediate_activation ) (d_i) 
    o       = Dense( config["original_dim"] ) (dec)

    decoder = Model(d_i, o, name='decoder') ## Instantiate decoder
     
    # Instantiate VAE
    vae_outputs = decoder (encoder(i)[2])               # type: ignore
    vae         = Model(i, vae_outputs, name='vae')
    print(f"Instantiated VAE ({time.time()-t}s)")
    t = time.time()

    # Define optimizer
    if config["solver"] == "nadam":
        optimizer = Nadam( config["learning_rate"] )
    print(f"Optimizer defined ({time.time()-t}s)")
    t = time.time()

    
    # Compile VAE
    loss_function = partial(kl_reconstruction_loss, mu=mu, sigma=sigma, original_dim=config["original_dim"])
    print(f"Loss function defined ({time.time()-t}s)")
    vae.compile(optimizer=optimizer, loss=loss_function)
    print(f"Complied ({time.time()-t}s)")
    
    return vae


class FIA_VAE_hptune:
    def __init__(self, X, test_size:float, configuration_space:ConfigurationSpace, model_builder, model_args):
        self.configuration_space = configuration_space
        self.model_builder = model_builder
        self.model_args = model_args
        self.training_data, self.test_data = train_test_split(X, test_size=test_size)

    def train(self, config: Configuration, seed: int = 0, budget: int = 25) -> float:
            t = time.time()
            keras.utils.set_random_seed(seed)
            model = self.model_builder(config=config)
            print(f"Model built. ({time.time()-t}s)")
            t = time.time()

            # Fit
            callback = keras.callbacks.EarlyStopping(monitor='loss', patience=100)	# Model will stop if no improvement
            model.fit(self.training_data, self.training_data, epochs=int(budget), verbose=0, callbacks=[callback])
            print(f"Model fitted. ({time.time()-t}s)")
            t = time.time()

            # Evaluation
            val_loss = model.evaluate(self.test_data,  self.test_data, verbose=0)
            print(f"Model evaluated ({time.time()-t}s)")
            t = time.time()
            keras.backend.clear_session()
            print(f"Session cleared. ({time.time()-t}s)")
                 
            return val_loss


# Final VAE
class FIA_VAE():

    def __init__(self,
                input_shape, intermediate_neurons:int, latent_dim:int,       # Shape
                kl_beta:float=1e-2, learning_rate:float=1e-3, intermediate_dropout:float=0.5, input_dropout:float=0.5,
                intermediate_activation=LeakyReLU()):                     # Rates
        
        self.input_shape = input_shape
        self.intermediate_neurons = intermediate_neurons
        self.latent_dim = latent_dim
        self.kl_beta = kl_beta
        self.learning_rate = learning_rate
        self.input_dropout = input_dropout
        self.intermediate_dropout = intermediate_dropout
        self.intermediate_activation = intermediate_activation
        
        # # =================
        # # Encoder
        # # =================
 
        self.input      = Input(shape=(self.input_shape,), name='encoder_input')
        self.input      = Dropout(self.intermediate_dropout) (self.input)
        self.enc        = Dense(self.intermediate_neurons, activation=self.intermediate_activation) (self.input)                          
        self.mu         = Dense(self.latent_dim, name='latent_mu') (self.enc)
        self.sigma      = Dense(self.latent_dim, name='latent_sigma') (self.enc)

        self.z          = Lambda(self.sample_z, output_shape=(self.latent_dim, ), name='z')([self.mu, self.sigma])      # Use reparameterization trick 

        self.encoder = Model(self.input, [self.mu, self.sigma, self.z], name='encoder')                                 # Instantiate encoder

        # =================
        # Decoder
        # =================

        # Definition
        self.decoder_input  = Input(shape=(self.latent_dim, ), name='decoder_input')
        self.dec            = Dense(self.intermediate_neurons, activation=self.intermediate_activation) (self.decoder_input)
        self.output  = Dense(self.input_shape) (self.dec )

        self.decoder = Model(self.decoder_input, self.output, name='decoder')                                           # Instantiate decoder

        # =================
        # VAE
        # =================

        # Instantiate VAE
        self.vae_outputs = self.decoder(self.encoder(self.input)[2])            # type: ignore
        self.vae         = Model(self.input, self.vae_outputs, name='vae')

        # Define optimizer
        self.optimizer = Nadam(learning_rate=self.learning_rate)

        # Compile VAE
        self.vae.compile(optimizer=self.optimizer, loss=self.kl_reconstruction_loss, metrics = ['mse'])
    
    
    def train(self, train_data, val_data, n_epochs, batch_size, verbosity=1):
        self.vae.fit(train_data, train_data,        	             # type: ignore
                     epochs = n_epochs, 
                     batch_size = batch_size, 
                     validation_data = (val_data, val_data),
                     verbose = verbosity)                            # type: ignore
    
    def encode(self, data):
        return self.encoder.predict(data)[2]                         # type: ignore
    
    def encode_mu(self, data):
        return self.encoder.predict(data)[0]                         # type: ignore
    
    def decode(self, data):
        return self.decoder.predict(data)                            # type: ignore
    
    def reconstruct(self, data):
        return self.decode(self.encode(data))
    
    def save_model(self, save_folder, suffix:str=""):
        self.vae.save(os.path.join(save_folder, f'VAE{suffix}.h5'))                  # type: ignore
        self.encoder.save(os.path.join(save_folder, f'VAE_encoder{suffix}.h5'))      # type: ignore
        self.decoder.save(os.path.join(save_folder, f'VAE_decoder{suffix}.h5'))      # type: ignore
        
    
    def load_vae(self, save_path):
        # The two functions below have to be redefined for the loading
        # of the model. They cannot be methods of the mtVAE class for
        # some reason.
        # https://github.com/keras-team/keras/issues/13992
                       
        self.vae = keras.models.load_model(save_path)
        self.vae.compile(optimizer=self.optimizer,                       # type: ignore
                         custom_objects={'sample_z': self.sample_z}, 
                         loss=self.kl_reconstruction_loss, 
                         metrics = ['mse'])
        
    def load_encoder(self, save_path):
        self.encoder = keras.models.load_model(save_path)
        
    def load_decoder(self, save_path):
        self.decoder = keras.models.load_model(save_path)

    def sample_z(self, args):
        """
        Define sampling with reparameterization trick
        """
        mu, sigma = args
        batch     = backend.shape(mu)[0]
        dim       = backend.shape(mu)[1]
        eps       = backend.random_normal(shape=(batch, dim))
        return mu + backend.exp(sigma / 2) * eps
    
    def kl_reconstruction_loss(self, true, pred):
        """
        Kullback-Leibler + Reconstruction loss
        """
        # Reconstruction loss
        reconstruction_loss = mse(true, pred) * self.input_shape

        # KL divergence loss
        kl_loss = 1 + self.sigma - backend.square(self.mu) - backend.exp(self.sigma)
        kl_loss = backend.sum(kl_loss, axis=-1)
        kl_loss *= -0.5

        # Total loss = mean(rec + scaler * KL divergence loss )
        return backend.mean(reconstruction_loss + self.kl_beta * kl_loss)