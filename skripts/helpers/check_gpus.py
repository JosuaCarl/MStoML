#!/usr/bin/env python3
#SBATCH --job-name Test_GPU
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 1

import os
import argparse

parser = argparse.ArgumentParser(prog='VAE_smac_run',
                                     description='Hyperparameter tuning for Variational Autoencoder with SMAC')
parser.add_argument('-b', '--backend', required=True)
args = parser.parse_args()

os.environ["KERAS_BACKEND"] = args.backend
import keras
import numpy as np
from GPUtil import showUtilization, getAvailable
import torch
import tensorflow as tf


print(f"Available GPUs: {getAvailable(limit=10)}")
showUtilization(all=True)
print()

print("Torch GPU available: ", torch.cuda.is_available())
print("Tensorflow devices found: ", tf.config.list_physical_devices())

data = np.random.rand(20,1000)
target = np.random.randint(0, 2, size=20)

print(f"Using backend: {keras.backend.backend()}\n")
model = keras.Sequential([keras.layers.Input(shape=(data.shape[1],)),
                          keras.layers.Dense(units=50000, activation="selu"),
                          keras.layers.Dense(units=1, activation="sigmoid")])

if torch.cuda.is_available() and keras.backend.backend() == "torch":
    torch.cuda.set_device(torch.device("cuda:0"))
    model = model.to("cuda")
    data = torch.tensor(data).to("cuda")
    target = torch.tensor(target).to("cuda")

print()
model.summary()

model.compile(optimizer="nadam", loss=keras.losses.binary_crossentropy, metrics=["accuracy"])
print("\nAfter compiliation:")
showUtilization(all=True)
history = model.fit(data, target, verbose=2, epochs=5)
print("\nAfter training:")
showUtilization(all=True)