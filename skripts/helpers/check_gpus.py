#!/usr/bin/env python3
#SBATCH --job-name Test_GPU
#SBATCH --partition gpu-a30
#SBATCH --nodes 4
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 1
#SBATCH --gres gpu:1,gpu:2

from GPUtil import showUtilization, getAvailable

print(f"Available GPUs: {getAvailable(limit=10)}")
showUtilization(all=True)

import torch
import tensorflow as tf

print("Torch found: ", torch.cuda.is_available())
print("Tensorflow found: ", tf.config.list_physical_devices())
