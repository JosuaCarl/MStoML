#!/usr/bin/env python3
#SBATCH --job-name VAE_training
#SBATCH --time 24:00:00
#SBATCH --mem 400G
#SBATCH --nodes 2
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 1

# available processors: cpu1, cpu2-hm, gpu-a30

import sys
import os
import argparse
from pathlib import Path
import time
import datetime
import numpy as np
import pandas as pd

from typing import Union
from ConfigSpace import Configuration, ConfigurationSpace

sys.path.append( '..' )
from helpers.normalization import *
from helpers.pc_stats import *


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


last_timestamp = time.time()
step = 0
runtimes = {}

def main():
    """
    One-time traing of Variational Autoencoder

    Example:
        The  following extracts data from ../data and uses ../runs as a directory to save to.
        It will run the model on the backend tensorflow and use gpu.
        It has the name 1 to differentiate it from other runs.
        It uses a batch_size of 16 and trains for 10000 epochs.
        It will construct a new model from the condiguration space, train, test and plot it.
        Its verbosity is set to 1, meaning surface level output.
        ```
        vae.py -d "../data" -r "../runs" -b "tensorflow" -c "gpu" -n "1" -bat 16 -e 10000 -s "new" "train" "test" "plot" -v 1
        ```
    """
    data_dir, run_dir = [os.path.normpath(os.path.join(os.getcwd(), d)) for d in  [args.data_dir, args.run_dir]]
    backend_name = args.backend
    computation = args.computation
    gpu = computation == "gpu"
    name = args.name if args.name else None
    project = f"vae_{backend_name}_{computation}_{name}" if name else f"vae_{backend_name}_{computation}"
    steps = args.steps
    verbosity =  args.verbosity if args.verbosity else 0
    outdir = Path(os.path.normpath(os.path.join(run_dir, project)))

    print(f"Using backend: {keras.backend.backend}")
    if verbosity > 0 and gpu:
        print_available_gpus()
        if "tensorflow" in backend_name:
            print("Available GPUs: ", tf.config.list_physical_devices('GPU'))
        if "torch" in backend_name:
            print("GPU available: ", torch.cuda.is_available())
            
    time_step(message="Setup loaded", verbosity=verbosity, min_verbosity=1)

    data = read_data(data_dir, verbosity=verbosity)

    batch_size = args.batch_size if args.batch_size else None
    epochs = args.epochs

    time_step("Start", verbosity=verbosity, min_verbosity=2)
    keras.utils.set_random_seed( 42 )

    if "new" in steps:
        config_space = ConfigurationSpace(
                {'input_dropout': 0.1, 'intermediate_activation': "relu", 'intermediate_dimension': 500,
                'intermediate_layers': 4, 'latent_dimension': 10, 'learning_rate': 0.001,
                'original_dim': 825000, 'solver': 'nadam'}
            )
        config = config_space.get_default_configuration()
    
        model = FIA_VAE(config)
        if verbosity >= 3:
            model.vae.summary()
            print_utilization(gpu=gpu)
    else:
        model = keras.saving.load_model(os.path.join(outdir, f"vae_{backend_name}_{computation}_{name}.keras"),
                                            custom_objects=None, compile=True, safe_mode=True)
        model.load_weights(os.path.join(outdir, f"vae_{backend_name}_{computation}_{name}.weights.h5"))
    
    time_step("Model built", verbosity=verbosity, min_verbosity=2)

    callbacks = []
    if verbosity >= 2:
        log_dir = os.path.join(outdir,  datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        callbacks.append( TensorBoard(log_dir=log_dir, histogram_freq=100, write_graph=True, write_images=True, update_freq='epoch') )

    if "train" in steps:
        model.fit(x=data, y=data, validation_split=0.2,
                  batch_size=batch_size, epochs=epochs,
                  callbacks=callbacks, verbose=verbosity)

        if verbosity >= 3:
            print("After training utilization:")
            print_utilization(gpu=gpu)
        time_step("Model trained", verbosity=verbosity, min_verbosity=2)
        
        model.save_weights( os.path.join(outdir, f"vae_{backend_name}_{computation}_{name}.weights.h5"), overwrite=True )
        model.save( os.path.join(outdir, f"vae_{backend_name}_{computation}_{name}.keras"), overwrite=True )
        time_step("Model trained", verbosity=verbosity, min_verbosity=2)        

    if "test" in steps:
        loss, recon_loss, kl_loss = model.evaluate(data, data,
                                                    batch_size=batch_size, verbose=verbosity, callbacks=callbacks)
        print({"Loss: ": loss, "Reconstruction loss: ": recon_loss, "Kullback-Leibler loss: ": kl_loss})
        time_step("Model evaluated", verbosity=verbosity, min_verbosity=2)
        
    if "predict" in steps:
        prediction = model.predict(data, batch_size=batch_size, verbose=verbosity, callbacks=callbacks)
        print(prediction)

    if "plot" in steps:
        keras.utils.plot_model( model,
                                to_file=os.path.join(outdir, f"vae_{backend_name}_{computation}_{name}.png"),
                                show_shapes=True, show_dtype=False, show_layer_names=True,
                                rankdir="TB", expand_nested=True, dpi=600,
                                show_layer_activations=True, show_trainable=True )



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

@keras.saving.register_keras_serializable(package="FIA_VAE")
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
    """
    A variational autoencoder for flow injection analysis
    """
    def __init__(self, config:Union[Configuration, dict]):
        super().__init__()
        self.config             = config
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
                                              [ Dense(config["original_dim"], activation="relu") ] , name="Decoder")

        # Loss trackers
        self.reconstruction_loss    = metrics.Mean(name="reconstruction_loss")
        self.kl_loss                = metrics.Mean(name="kl_loss")
        self.loss_tracker           = metrics.Mean(name="loss")

        # Define optimizer
        self.optimizer = get_solver( config["solver"] )( config["learning_rate"] )

        # Compile VAE
        self.compile(optimizer=self.optimizer, loss=self.kl_reconstruction_loss)

    @property
    def metrics(self):
        return [self.loss_tracker, self.reconstruction_loss, self.kl_loss]
    
    def get_config(self):
        return {"config": dict(self.config)}

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
    
    @keras.saving.register_keras_serializable(package="FIA_VAE")
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
    parser = argparse.ArgumentParser(prog='VAE_smac_run',
                                     description='Hyperparameter tuning for Variational Autoencoder with SMAC')
    parser.add_argument('-d', '--data_dir', required=True)
    parser.add_argument('-r', '--run_dir', required=True)
    parser.add_argument('-b', '--backend', required=True)
    parser.add_argument('-c', '--computation', required=True)
    parser.add_argument('-n', '--name', required=False)
    parser.add_argument('-bat', '--batch_size', type=int, required=False)
    parser.add_argument('-e', '--epochs', type=int, required=True)
    parser.add_argument('-s', '--steps', nargs="+", required=True)
    parser.add_argument('-v', '--verbosity', type=int, required=False)
    args = parser.parse_args()
    
    os.environ["KERAS_BACKEND"] = args.backend

    main()
