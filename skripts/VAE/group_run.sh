sbatch --job-name=VAE_tuning_mae --partition=cpu2-hm --nodelist=hm024 --time=1-00:00:00 --output=../../runs/VAE/hyperparameter_optimization/mae.out VAE_smac.py -d "../../runs/FIA/Sampled_Coms" -r "../../runs/VAE/hyperparameter_optimization" -b "tensorflow" -c "cpu" -p "float32" -n "2" --bat 100 -v 1 -l "mae"
sbatch --job-name=VAE_tuning_cos --partition=cpu2-hm --nodelist=hm024 --time=1-00:00:00 --output=../../runs/VAE/hyperparameter_optimization/cos.out VAE_smac.py -d "../../runs/FIA/Sampled_Coms" -r "../../runs/VAE/hyperparameter_optimization" -b "tensorflow" -c "cpu" -p "float32" -n "2" --bat 100 -v 1 -l "cosine"
sbatch --job-name=VAE_tuning_maecos --partition=cpu2-hm --nodelist=hm024 --time=1-00:00:00 --output=../../runs/VAE/hyperparameter_optimization/maecos.out VAE_smac.py -d "../../runs/FIA/Sampled_Coms" -r "../../runs/VAE/hyperparameter_optimization" -b "tensorflow" -c "cpu" -p "float32" -n "2" --bat 100 -v 1 -l "mae+cosine"
sbatch --job-name=VAE_tuning_se --partition=cpu2-hm --nodelist=hm024 --time=1-00:00:00 --output=../../runs/VAE/hyperparameter_optimization/se.out VAE_smac.py -d "../../runs/FIA/Sampled_Coms" -r "../../runs/VAE/hyperparameter_optimization" -b "tensorflow" -c "cpu" -p "float32" -n "2" --bat 100 -v 1 -l "spectral_entropy"