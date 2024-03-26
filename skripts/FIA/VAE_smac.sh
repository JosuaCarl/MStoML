#!/usr/bin/bash
conda activate MStoML
python VAE_smac.py -d "../../runs/FIA/comm8/oms" -r "../../runs/VAE/hyperband_optimization"
echo "Done"