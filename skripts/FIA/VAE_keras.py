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
from keras import layers
from keras.layers import Input, Dense, Dropout
from keras.losses import mse
from keras import optimizers
from keras import activations
import keras.backend as backend
import keras

from ConfigSpace import Configuration, ConfigurationSpace

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

from skripts.helpers.pc_stats import *

print("Available GPUs: ", tf.config.list_physical_devices('GPU'))

# Helpers
def get_activation_function(activation_function:str):
    """
    Convert an activation function string into a pytorch function

    Args:
        activation_function (str): Activation function in string representation 

    Returns:
        Activation function as a pytorch.nn class
    """
    if activation_function == "relu":
        return activations.relu()
    elif activation_function == "leakyrelu":
        return layers.LeakyReLU()
    elif activation_function == "selu":
        return activations.selu()
    elif activation_function == "tanh":
        return activations.tanh()
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
        return optimizers.legacy.Adam
    elif solver == "nadam":
        return optimizers.legacy.Nadam
    elif solver == "adamw":
        return optimizers.AdamW
    else:
        raise(ValueError(f"{solver} is not defined as a valid solver."))


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
        activation_fun  = config["intermediate_activation"]
        
        # Encoder
        self.input      = Input(shape=(config["original_dim"],), name='encoder_input')
        self.input      = Dropout( config["input_dropout"] ) (self.input)
        self.im_enc    = Dense(im_dim , activation=activation_fun) (self.input)
        for i in range(1, im_layers):                                                              # Successive halfing of layers
            if im_dim // 2**i <= config["latent_dimension"]:
                im_layers = i 
                break
            self.im_enc = Dense( im_dim // 2**i, activation=activation_fun ) (self.im_enc)
        self.mu         = Dense(config["latent_dimension"], name='latent_mu') (self.im_enc)
        self.sigma      = Dense(config["latent_dimension"], name='latent_sigma') (self.im_enc)
        self.z          = Sampling() ([self.mu, self.sigma])                                       # Use reparameterization trick 
        self.encoder = Model(self.input, [self.mu, self.sigma, self.z], name='encoder')            # Instantiate encoder

        # Decoder
        self.decoder_input  = Input(shape=(config["latent_dimension"], ), name='decoder_input')
        for i in reversed(range(1, im_layers)):
            prev_layer = self.decoder_input if i == im_layers-1 else self.im_dec
            self.im_dec =  Dense( im_dim // 2**i, activation=activation_fun) (prev_layer)
        self.im_dec = Dense(im_dim, activation=activation_fun) (self.im_dec)
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
        self.vae.compile(optimizer=self.optimizer, loss=self.kl_reconstruction_loss,
                         metrics = [self.loss, self.reconstruction_loss, self.kl_loss])

    @property
    def metrics(self):
        return [
            self.loss,
            self.reconstruction_loss,
            self.kl_loss,
        ]

    def train(self, train_data, val_data, n_epochs, batch_size, verbosity=1):
        self.vae.fit(train_data, train_data,
                     epochs = n_epochs, 
                     batch_size = batch_size, 
                     validation_data = (val_data, val_data),
                     verbose = verbosity)
    
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
    

class FIA_VAE_hptune:
    def __init__(self, X, test_size:float, configuration_space:ConfigurationSpace, model_builder,
                 verbosity:int=0):
        self.configuration_space = configuration_space
        self.model_builder = model_builder
        self.training_data, self.test_data = train_test_split(X, test_size=test_size)
        self.verbosity = verbosity

    def train(self, config: Configuration, seed: int = 0, budget:int=25) -> float:
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
        keras.utils.set_random_seed(seed)

        # Definition
        model = self.model_builder(config)
        if self.verbosity > 1:
            if self.verbosity > 2:
                model.summary()
                print_utilization()
            print(f"Model built in {time.time()-t}s")
            t = time.time()

        # Fitting
        callback = keras.callbacks.EarlyStopping(monitor='loss', patience=100)	# Model will stop if no improvement
        model.fit(self.training_data, self.training_data, epochs=int(budget), verbose=0, callbacks=[callback])
        if self.verbosity > 1:
            if self.verbosity > 2:
                print("After training utilization:")
                print_utilization()
            print(f"Model trained in {time.time()-t}s")
            t = time.time()

        # Evaluation
        loss, reconstruction_loss, kl_loss = model.evaluate(self.test_data,  self.test_data, verbose=0)
        if self.verbosity > 1:
            print(f"Model evaluated in {time.time()-t}s")
        
        # Clearing model parameters
        keras.backend.clear_session()
        if self.verbosity > 1:
            print(f"Session cleared in ({time.time()-t}s)")
                
        return loss