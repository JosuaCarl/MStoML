#!/usr/bin/env python3
"""
Definition and training/testing of a variational Autoencoder for Flow-injection analysis.
"""

#SBATCH --job-name VAE_training
#SBATCH --mem 400G
#SBATCH --nodes 1
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
import json

from typing import Union
from ConfigSpace import Configuration, ConfigurationSpace

sys.path.append( '..' )
from helpers.normalization import *
from helpers.pc_stats import *
import mlflow

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import keras
from keras import Model, Sequential
from keras.layers import Input, Dense, Dropout, GaussianNoise
from keras import backend, ops, layers, activations, metrics, losses, optimizers

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

import tensorflow as tf

last_timestamp = time.time()
step = 0
runtimes = {}

def main(args):
    """
    Training of Variational Autoencoder

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
    data_dir, run_dir, tune_dir= [os.path.normpath(os.path.join(os.getcwd(), d)) for d in  [args.data_dir, args.run_dir, args.tune_dir]]
    backend_name = args.backend
    computation = args.computation
    gpu = computation == "gpu"
    name = args.name if args.name else None
    project = f"vae_{backend_name}_{computation}_{name}" if name else f"vae_{backend_name}_{computation}"
    batch_size = args.batch_size if args.batch_size else None
    epochs = args.epochs
    steps = args.steps
    verbosity =  args.verbosity if args.verbosity else 0
    outdir = Path(os.path.normpath(os.path.join(run_dir, project)))
    precision = args.precision if args.precision else "float32"
    validation_split = 0.2

    print(f"Using backend: {keras.backend.backend()}")
    if verbosity > 0 and gpu:
        print_available_gpus()
        if "tensorflow" in backend_name:
            print("Tensorflow devices found: ", tf.config.list_physical_devices())
            print(f"Keras Devices found: {keras.distribution.list_devices(device_type=None)}")
        elif "torch" in backend_name:
            print("GPU available: ", torch.cuda.is_available())
    
    keras.backend.set_floatx(precision)

    time_step(message="Setup loaded", verbosity=verbosity, min_verbosity=1)

    data = read_data(data_dir, verbosity=verbosity)
    print(f"Shape before batch pruning: {data.shape}")
    if batch_size and batch_size < len(data):
        drop_last = len(data) % batch_size
        drop_split = (len(data) - drop_last) % (1 / validation_split)
        data = data.iloc[:-int(drop_last + drop_split)]
    if backend.backend() == "torch":
        data = torch.tensor( data.to_numpy() ).to( model.device )
    print(f"Shape after pruning: {data.shape}")

    time_step("Data read", verbosity=verbosity, min_verbosity=2)

    
    keras.utils.set_random_seed( 42 )
    previous_history = []
    if "new" in steps:
        if tune_dir:
            with open(os.path.join(tune_dir,f"smac_{project}", "best_config.json"), "r") as infile:
                config_dict = json.load(infile)
                config_space = ConfigurationSpace( config_dict )
        else:
            config_space = ConfigurationSpace(
                {
                    'input_dropout': 0.08235615567775992,
                    'intermediate_activation': 'silu',
                    'intermediate_dimension': 1779,
                    'intermediate_layers': 8,
                    'kld_weight': 1.1957167373762512,
                    'latent_dimension': 12,
                    'learning_rate': 2.8519767469131045e-05,
                    'original_dim': 825000,
                    'reconstruction_loss_function': 'ae+cosine',
                    'solver': 'nadam',
                    'stdev_noise': 1.4547834681520132e-11,
                    'tied': 0,
                    }
                )
        config = config_space.get_default_configuration()
    
        model = FIA_VAE(config)
        if verbosity >= 3:
            model.summary()
            print_utilization(gpu=gpu) 
    else:
        model = keras.saving.load_model(os.path.join(outdir, f"vae_{backend_name}_{computation}_{name}.keras"), safe_mode=True)
        model.load_weights(os.path.join(outdir, f"vae_{backend_name}_{computation}_{name}.weights.h5"))
        if os.path.join(outdir, f"vae_{backend_name}_{computation}_{name}.history.tsv"):
            previous_history = pd.read_csv( os.path.join(outdir, f"vae_{backend_name}_{computation}_{name}.history.tsv"), sep="\t" )
    
    time_step("Model built", verbosity=verbosity, min_verbosity=2)


    callbacks = []
    if "train" in steps:
        callbacks.append( keras.callbacks.ModelCheckpoint( filepath=str(outdir) + f"/vae_{backend_name}_{computation}_{name}" + "_best.keras",
                                                           save_best_only=True, monitor="val_loss",
                                                           verbose=verbosity ) )

        mlflow.set_tracking_uri(Path(os.path.join(outdir, "mlruns")))
        mlflow.set_experiment(f"FIA_VAE")
        mlflow.autolog(log_datasets=False, log_models=False, silent=verbosity < 2)
        with mlflow.start_run(run_name=f"fia_vae"):
            history = model.fit(data, data, validation_split=validation_split,
                                batch_size=batch_size, epochs=epochs,
                                callbacks=callbacks, verbose=verbosity)
            mlflow.log_params(model.config)
        
        if verbosity >= 3:
            print("After training utilization:")
            print_utilization(gpu=gpu)
        time_step("Model trained", verbosity=verbosity, min_verbosity=2)
        
        model.save_weights( os.path.join(outdir, f"vae_{backend_name}_{computation}_{name}.weights.h5"), overwrite=True )
        model.save( os.path.join(outdir, f"vae_{backend_name}_{computation}_{name}.keras"), overwrite=True )
        
        history_df = pd.DataFrame(history.history)
        if previous_history:
            history_df = pd.concat([previous_history, history_df], ignore_index=True)
        history_df.to_csv( os.path.join(outdir, f"vae_{backend_name}_{computation}_{name}.history.tsv"), sep="\t")

        time_step("Model trained", verbosity=verbosity, min_verbosity=2)        

    if "test" in steps:
        loss, recon_loss, kl_loss = model.evaluate(data, data,
                                                    batch_size=batch_size, verbose=verbosity, callbacks=callbacks)
        print({"Loss: ": loss, "Reconstruction loss: ": recon_loss, "Kullback-Leibler loss: ": kl_loss})
        time_step("Model evaluated", verbosity=verbosity, min_verbosity=2)
        
    if "predict" in steps:
        prediction = model.predict(data, batch_size=batch_size, verbose=verbosity, callbacks=callbacks)
        time_step("Prediction made", verbosity=verbosity, min_verbosity=2)
        prediction_df = pd.DataFrame(prediction)
        prediction_df.to_csv( os.path.join(outdir, f"vae_{backend_name}_{computation}_{name}.prediction.tsv"), sep="\t" )

    if "plot" in steps:
        keras.utils.plot_model( model,
                                to_file=os.path.join(outdir, f"vae_{backend_name}_{computation}_{name}.png"),
                                show_shapes=True, show_dtype=False, show_layer_names=True,
                                rankdir="TB", expand_nested=True, dpi=600,
                                show_layer_activations=True, show_trainable=True )



def read_data(data_path:str, verbosity:int=0):
    """
    Read in the data from a data_matrix and normalize it according to total ion counts.

    :param data_path: Directory with "data_matrix.tsv" file. The rows must represent m/z bins, the columns different samples
    :type data_path: str
    :param verbosity: verbosity of process, defaults to 0
    :type verbosity: int, optional
    :return: matrix with total ion count (TIC) normalized data (transposed)
    :rtype: DataFrame
    """
    if data_path.endswith("tsv"):
        binned_dfs = pd.read_csv( data_path, sep="\t", index_col="mz")
    elif data_path.endswith("feather"):
        binned_dfs = pd.read_feather( data_path )
    elif data_path.endswith("parquet"):
        binned_dfs = pd.read_parquet( data_path )
    else:
        binned_dfs = pd.read_csv( os.path.join(data_path, "data_matrix.tsv"), sep="\t", index_col="mz")
    binned_dfs[:] =  total_ion_count_normalization(binned_dfs)

    X = binned_dfs.transpose()
    time_step(message="Data loaded", verbosity=verbosity, min_verbosity=1)
    return X


def time_step(message:str, verbosity:int=0, min_verbosity:int=1):
    """
    Saves the time difference between last and current step

    :param message: printed message at this timestep
    :type message: str
    :param verbosity: verbosity of process, defaults to 0
    :type verbosity: int, optional
    :param min_verbosity: minimal verbosity to display output, defaults to 1
    :type min_verbosity: int, optional
    """
    global last_timestamp
    global step
    global runtimes
    runtimes[f"{step}: {message}"] = time.time() - last_timestamp
    if verbosity >= min_verbosity: 
        print(f"{message} ({runtimes[f'{step}: {message}']}s)")
    last_timestamp = time.time()
    step += 1


@keras.saving.register_keras_serializable(package="FIA_VAE")
def spectral_entropy(spectrum):
    """
    Computes the spectral entropy for a spectra $S_i \in S$ with elements $s_i \in S_i$: $-\sum (s_i * ln(s_i))$:

    :param spectrum: spectrum input
    :type spectrum: Keras tensor or similar
    :return: spectral entropy of spectrum
    :rtype: Keras tensor or similar
    """    
    spectrum = ops.normalize(spectrum, order=1) # Total count normalization
    return -ops.sum( ops.multiply( spectrum, ops.log( ops.add(spectrum, 1e-32) ) ), axis=-1 )

@keras.saving.register_keras_serializable(package="FIA_VAE")
def mean_spectral_entropy_divergence(y_true, y_pred):
    """
    Computes the mean spectral entropy divergence over a given list of spectra.

    :param y_true: True target values
    :type y_true: Keras tensor or similar
    :param y_pred: Predicted target values
    :type y_pred: Keras tensor or similar
    :return: Mean spectral entropy divergence (MSED) between y_true and y_pred
    :rtype: Keras tensor or similar
    """    
    y_comb = ops.add(y_true, y_pred)
    return ops.mean( ops.abs(                     # floating point errors can lead to small negative values
                ops.divide(
                    ops.subtract( ops.multiply(2, spectral_entropy( y_comb ) ) ,
                                  ops.add( spectral_entropy( y_true ), spectral_entropy( y_pred ) ) ),
                    ops.log(4) ) ) )

@keras.saving.register_keras_serializable(package="FIA_VAE")
class Sampling(layers.Layer):
    """
    Sampling class as a sublass of keras layer.

    :param layers: Layer type
    :type layers: keras.layers.Layer
    """    

    def call(self, inputs):
        """
        Uses (z_mean, z_log_var) to sample z, the vector encoding a digit.

        :param inputs: Mean tensor $z_\mu$/z_mean and Logarithmic variance tensor $\ln z_\sigma$/z_log_var.
        :type inputs: Keras tensor or similar
        :return: Sampled value within a normal distribution, specified by the input parameters.
        :rtype: Keras tensor or similar
        """        
        z_mean, z_log_var = inputs
        z_mean_shape = ops.shape(z_mean)
        batch   = z_mean_shape[0]
        dim     = z_mean_shape[1]
        epsilon = keras.random.normal(shape=(batch, dim))
        return ops.multiply(ops.add(z_mean, ops.exp(0.5 * z_log_var)), epsilon)
    
@keras.saving.register_keras_serializable(package="FIA_VAE")
class DenseTied(keras.layers.Layer):
    """
    A Layer Tied to another Dense Layer with shared weights.

    :param keras: Subclass of layer to share weights and biases with another Dense Layer
    :type keras: keras.layers.Layer
    """    
    def __init__(self, tie, activation=None, **kwargs):
        """
        Initiation of tied Layer

        :param tie: Layer which this one is tied to.
        :type tie: keras.layers.Dense
        :param activation: Activation function, defaults to None
        :type activation: Function, parsable by keras.activation.get, optional
        """        
        self.tie = tie
        self.activation = keras.activations.get(activation)
        super().__init__(**kwargs)
    def build(self, batch_input_shape):
        """
        Build the Dense Layer for the respective batch_input_shape. The input is ingored, but needs
        to be present for Keras.

        :param batch_input_shape: Shape of the input, dependent on batch size.
        :type batch_input_shape: Keras tensor or similar
        """        
        self.biases = self.add_weight(name="bias", initializer="zeros", shape=[ops.shape(self.tie.input)[-1]])
        self.kernel = ops.transpose(self.tie.kernel)
        super().build(batch_input_shape)
    def call(self, inputs):
        """
        Call the layer

        :param inputs: Input tensor
        :type inputs: Keras tensor or similar
        :return: Transform layer with weights, activation function and bias.
        :rtype: Keras tensor or similar
        """        
        z = ops.matmul(inputs, self.kernel)
        return self.activation(z + self.biases)


@keras.saving.register_keras_serializable(package="FIA_VAE")
class FIA_VAE(Model):
    """
    A variational autoencoder for flow injection analysis

    :param Model: Suptype of keras.Model
    :type Model: keras.Model
    """    
    def __init__(self, config:Union[Configuration, dict]):
        """
        Initalize the variational autoencoder. 
        The configuration may contain: The original dimension of the input, whether the VAE is tied,
        the number of intermediate layers, the activation function that is used for the dense layers,
        a dropout rate for the input, gaussian noise standard deviation for the input, the latent and
        intermediate stating dimension, the solver/optimizer with a learning rate, the weight of the
        Kullback-Leibler distance in the loss function, and the reconstruction loss function.

        :param config: Configuration of Autoencoder.
        :type config: Union[ConfigSpace.Configuration, dict]
        """        
        super().__init__()
        self.device             = "cuda" if torch.cuda.is_available() else "cpu"
        self.config             = dict(config)
        self.tied               = config["tied"] if "tied" in self.config else False
        intermediate_layers     = [i for i in range( config["intermediate_layers"] ) 
                                    if config["intermediate_dimension"] // 2**i > config["latent_dimension"]]
        activation_function     = self.get_activation_function( config["intermediate_activation"] )

        # Encoder (with sucessive halfing of intermediate dimension)
        self.dropout            = Dropout( config["input_dropout"] , name="dropout")
        self.noise              = GaussianNoise( config["stdev_noise"] )
        self.intermediate_enc   = Sequential ( [ Input(shape=(config["original_dim"],), name='encoder_input') ] +
                                               [ Dense( config["intermediate_dimension"] // 2**i,
                                                        activation=activation_function ) 
                                                for i in intermediate_layers] +
                                               [ Dense( config["latent_dimension"] ) ] , name="encoder_intermediate")

        self.mu_encoder         = Dense( config["latent_dimension"], name='latent_mu' )
        self.sigma_encoder      = Dense( config["latent_dimension"], name='latent_sigma' )
        self.z_encoder          = Sampling( name="latent_reparametrization" ) 

        # Decoder
        self.decoder            = Sequential( [ Input(shape=(config["latent_dimension"], ), name='decoder_input') ] +
                                              [ DenseTied( tie=self.intermediate_enc.get_layer(index=i+1),
                                                           activation=activation_function ) 
                                                if self.tied else
                                                Dense( config["intermediate_dimension"] // 2**i,
                                                       activation=activation_function )
                                               for i in reversed(intermediate_layers) ] +
                                              [ DenseTied(tie=self.intermediate_enc.get_layer(index=0), activation="relu")
                                                if self.tied else
                                                Dense(config["original_dim"], activation="relu") ] , name="Decoder")

        # Define optimizer
        self.optimizer              = self.get_solver( config["solver"] )( config["learning_rate"] )

        # Loss weight + trackers
        self.kld_weight             = config["kld_weight"] if "kld_weight" in dict(config) else 1.0
        self.reconstruction_loss_function = self.get_reconstruction_loss_function( config["reconstruction_loss_function"] )
        self.reconstruction_loss    = metrics.Mean(name="reconstruction_loss")
        self.kl_loss                = metrics.Mean(name="kl_loss")
        self.loss_tracker           = metrics.Mean(name="loss")

        # Compile VAE
        self.compile(optimizer=self.optimizer)

        # Config correction
        self.config["intermediate_layers"]  = len(intermediate_layers)

        if backend.backend() == "torch":
            self.to( self.device )

    def get_activation_function(self, activation_function:str):
        """
        Convert an activation function string into a keras function.

        :param activation_function: Activation function in string representation.
        :type activation_function: str
        :return: Activation function as keras activation.
        :rtype: keras.activations.Activation
        """        
        activation_functions = {"relu": activations.relu, "leaky_relu" : activations.leaky_relu,
                                "selu": activations.selu, "tanh": activations.tanh,
                                "silu": activations.silu, "mish": activations.mish}
        return activation_functions[activation_function]

    def get_reconstruction_loss_function(self, reconstruction_loss_function:str):
        """
        Convert an activation function string into a keras function

        :param reconstruction_loss_function: Reconstruction loss function in string representation
        :type reconstruction_loss_function: str
        :return: Reconstruction loss function as a lambda function or inherent Keras loss function
        :rtype: keras.losses.Loss
        """        
        reconstruction_loss_functions = {"mae": losses.mean_absolute_error,
                                         "mse": losses.mean_squared_error,
                                         "cosine": lambda y_true, y_pred: 1 + losses.cosine_similarity(y_true, y_pred),
                                         "mae+cosine": lambda y_true, y_pred:
                                            1 + losses.cosine_similarity(y_true, y_pred) + losses.mean_absolute_error(y_true, y_pred),
                                         "ae+cosine":lambda y_true, y_pred:
                                            1 + losses.cosine_similarity(y_true, y_pred) + ops.sum(ops.abs(y_true - y_pred)),
                                         "spectral_entropy": mean_spectral_entropy_divergence
                                         }
        return reconstruction_loss_functions[reconstruction_loss_function]
    
    def get_solver(self, solver:str):
        """
        Convert an solver string into a keras function.

        :param solver: Solver in string representation
        :type solver: str
        :return: Solver in Keras optimizer representation
        :rtype: keras.optimizers.Optimizer
        """        
        solvers = {"adam": optimizers.Adam, "nadam": optimizers.Nadam, "adamw": optimizers.AdamW}
        return solvers[solver]
    
    def kl_reconstruction_loss(self, y_true, y_pred):
        """
        Loss function for Kullback-Leibler + Reconstruction loss

        :param y_true: True values
        :type y_true: Keras tensor or similar
        :param y_pred: Predicted values
        :type y_pred: Keras tensor or similar
        :return: Loss = Kullback-Leibler * KLD weight + Reconstruction loss
        :rtype: Keras tensor or similar
        """
        reconstruction_loss = self.reconstruction_loss_function(y_true, y_pred)
        kl_loss = -0.5 * ops.sum( 1.0 + self.sigma - ops.square(self.mu) - ops.exp(self.sigma) )
        loss = reconstruction_loss + self.kld_weight * kl_loss
        
        return {"reconstruction_loss": reconstruction_loss, "kl_loss": kl_loss, "loss": loss}

    @property
    def metrics(self):
        """
        Metrics (Loss, Reconstruction loss, KL-loss), used for model evalutation.

        :return: Metrics
        :rtype: keras.metrics.Metric
        """        
        return [self.loss_tracker, self.reconstruction_loss, self.kl_loss]
    
    def get_config(self):
        """
        Return configuration pf VAE as a dictionary.

        :return: Configuration of VAE
        :rtype: dict
        """        
        return {"config": dict(self.config)}

    def call(self, data, training=False):
        """
        Encode and decode the data once.

        :param data: Input data
        :type data: Keras Tensor or similar
        :param training: Used to switch to training mode, enables dropout and noise, defaults to False
        :type training: bool, optional
        :return: Reconstructed Data
        :rtype: Keras Tensor or similar
        """        
        x = self.dropout(data, training=training)
        x = self.noise(data, training=training)
        return self.decode(self.encode(x))

    def encode(self, data):
        """
        Encode the data.

        :param data: Input data
        :type data: Keras Tensor or similar
        :return: Encoded Data
        :rtype: Keras Tensor or similar
        """        
        x = self.intermediate_enc(data)
        self.mu = self.mu_encoder(x)
        self.sigma = self.sigma_encoder(x)
        self.z = self.z_encoder( [self.mu, self.sigma] )
        return self.z
    
    def encode_mu(self, data):
        """
        Encode only the mean $\mu$ of the distribution.

        :param data: Input data
        :type data: Keras Tensor or similar
        :return: Encoded $\mu$
        :rtype: Keras Tensor or similar
        """        
        x = self.intermediate_enc(data)
        return self.mu_encoder(x)
    
    def decode(self, x):
        """
        Decode the latent space.

        :param x: Input tensor
        :type x: Keras Tensor or similar
        :return: Reconstructed representation of latent space
        :rtype: Keras Tensor or similar
        """        
        return self.decoder(x)

    def train_step(self, data):
        """
        Perform a single step in training.

        :param data: Input data
        :type data: Keras Tensor or similar
        :return: Loss
        :rtype: keras.losses.Loss
        """        
        x, y = data
        if backend.backend() == "torch":
            x, y = (x.to(self.device) , y.to(self.device))
            self.zero_grad()
            y_pred = self(x, training=True)
            loss = self.kl_reconstruction_loss(y, y_pred)
            loss.backward()

            trainable_weights = [v for v in self.trainable_weights]
            gradients = [v.value.grad for v in trainable_weights]

            with torch.no_grad():
                self.optimizer.apply(gradients, trainable_weights)

        elif backend.backend() == "tensorflow":
            with tf.GradientTape() as tape:
                y_pred = self(x, training=True)
                loss = self.kl_reconstruction_loss(y, y_pred)

            # Compute gradients
            trainable_vars = self.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)

            self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.reconstruction_loss.update_state( loss["reconstruction_loss"] )
        self.kl_loss.update_state( loss["kl_loss"] )
        self.loss_tracker.update_state( loss["loss"] )
        return loss
    
    def test_step(self, data):
        """
        Perform a single step in testing.

        :param data: Input data
        :type data: Keras Tensor or similar
        :return: Loss
        :rtype: keras.losses.Loss
        """        
        x, y = data
        y_pred = self(x, training=False)
        loss = self.kl_reconstruction_loss(y, y_pred)
        self.reconstruction_loss.update_state( loss["reconstruction_loss"] )
        self.kl_loss.update_state( loss["kl_loss"] )
        self.loss_tracker.update_state( loss["loss"] )
        return loss



if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='vae',
                                     description='Training a Variational Autoencoder')
    parser.add_argument('-d', '--data_dir', required=True)
    parser.add_argument('-r', '--run_dir', required=True)
    parser.add_argument('-t', '--tune_dir', required=True)
    parser.add_argument('-b', '--backend', required=True)
    parser.add_argument('-c', '--computation', required=True)
    parser.add_argument('-p', '--precision', required=False)
    parser.add_argument('-n', '--name', required=False)
    parser.add_argument('-bat', '--batch_size', type=int, required=False)
    parser.add_argument('-e', '--epochs', type=int, required=True)
    parser.add_argument('-s', '--steps', nargs="+", required=True)
    parser.add_argument('-v', '--verbosity', type=int, required=False)
    args = parser.parse_args()
    
    os.environ["KERAS_BACKEND"] = args.backend

    main(args)
