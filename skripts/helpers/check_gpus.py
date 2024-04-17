<<<<<<< HEAD
#!/usr/bin/env python3
#SBATCH --job-name Test_GPU
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 1

import os
import argparse
from GPUtil import showUtilization, getAvailable
import torch
import tensorflow as tf

print(f"Available GPUs: {getAvailable(limit=10)}")
showUtilization(all=True)

print("Torch available: ", torch.cuda.is_available())
print("Tensorflow found: ", tf.config.list_physical_devices())
print("Tensorflow available: ", tf.test.is_gpu_available())


parser = argparse.ArgumentParser(prog='VAE_smac_run',
                                     description='Hyperparameter tuning for Variational Autoencoder with SMAC')
parser.add_argument('-b', '--backend', required=False)
args = parser.parse_args()

os.environ["KERAS_BACKEND"] = args.backend if args.backend else "tensorflow"
import keras
import numpy as np

data = np.random.rand(20,100)
target = np.random.randint(0, 2, size=20)

model = keras.Sequential([keras.layers.Input(shape=(data.shape[1],)),
                          keras.layers.Dense(units=5, activation="selu"),
                          keras.layers.Dense(units=1, activation="sigmoid")])

model.compile(optimizer="nadam", loss=keras.losses.binary_crossentropy, metrics=["accuracy"])
history = model.fit(data, target, verbose=2, epochs=5)
print("After training:")
=======
#!/usr/bin/env python3
#SBATCH --job-name Test_GPU
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 1

import os
import argparse
from GPUtil import showUtilization, getAvailable
import torch
import tensorflow as tf

print(f"Available GPUs: {getAvailable(limit=10)}")
showUtilization(all=True)

print("Torch found: ", torch.cuda.is_available())
print("Tensorflow found: ", tf.config.list_physical_devices())

parser = argparse.ArgumentParser(prog='VAE_smac_run',
                                     description='Hyperparameter tuning for Variational Autoencoder with SMAC')
parser.add_argument('-b', '--backend', required=False)
args = parser.parse_args()

os.environ["KERAS_BACKEND"] = args.backend if args.backend else "tensorflow"
import keras
import numpy as np

data = np.random.rand(20,100)
target = np.random.randint(0, 2, size=20)


model = keras.Sequential([keras.layers.Input(shape=(data.shape[1],)),
                          keras.layers.Dense(units=5, activation="selu"),
                          keras.layers.Dense(units=1, activation="sigmoid")])

model.compile(optimizer="nadam", loss=keras.losses.binary_crossentropy, metrics=["accuracy"])
history = model.fit(data, target, verbose=2, epochs=5)
print("After training:")
>>>>>>> d3de9f4 (Local commit)
showUtilization(all=True)