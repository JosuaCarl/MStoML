#!/usr/bin/bash
python VAE_smac.py -d "../../runs/FIA/Com8_grown_together/oms" -r "../../runs/VAE/hyperband_optimization"
echo "Done"
