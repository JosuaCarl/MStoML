#!/usr/bin/env python3
#SBATCH --job-name VAE_tuning
#SBATCH --time 24:00:00
#SBATCH --mem 400G
#SBATCH --nodes 2
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 1

# available processors: cpu1, cpu2-hm, gpu-a30

# imports
import sys
import os
import time
import argparse
from tqdm import tqdm
from pathlib import Path
import pandas as pd

from ConfigSpace import Categorical, Configuration, ConfigurationSpace, Float, Integer, Constant, ForbiddenGreaterThanRelation
from smac import MultiFidelityFacade
from smac import Scenario
from smac.intensifier.hyperband import Hyperband
from smac.runhistory.dataclasses import TrialValue

# Doesn't work with Slurm, because __file__ variable is not copied
"""
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append( os.path.normcase(os.path.join( dir_path, '..' )))
print(os.path.normpath(os.path.join( dir_path, '..' )))
"""
sys.path.append("..")
from helpers.pc_stats import *
from FIA.FIA import *


# Argument parser
parser = argparse.ArgumentParser(prog='VAE_smac_run',
                                description='Hyperparameter tuning for Variational Autoencoder with SMAC')
parser.add_argument('-d', '--data_dir')
parser.add_argument('-r', '--run_dir')
parser.add_argument('-t', '--test_configuration')
parser.add_argument('-v', '--verbosity')
parser.add_argument('-f', '--framework')
parser.add_argument('-o', '--overwrite')
args = parser.parse_args()

from VAE.VAE_keras import *
from VAE.VAE_torch import *        

# Logging (time and steps)
last_timestamp = time.time()
step = 0
runtimes = {}

def main():
    """
    Hyperparameter optimization with SMAC3
    """
    data_dir, run_dir = [os.path.normpath(os.path.join(os.getcwd(), d)) for d in  [args.data_dir, args.run_dir]]
    test_configuration, overwrite, verbosity = (bool(args.test_configuration), bool(args.overwrite), int(args.verbosity))
    framework = args.framework
    outdir = Path(os.path.normpath(os.path.join(run_dir, f"smac_vae_{framework}")))
    time_step(message="Setup loaded", verbosity=verbosity)

    X = read_data(data_dir, verbosity=verbosity)

    configuration_space = ConfigurationSpace(seed=42)
    hyperparameters = [
        Constant(       "original_dim",             X.shape[1]),
        Float(          "input_dropout",            (0.0, 0.5), default=0.25),
        Integer(        "intermediate_layers",      (1, 5), default=2),
        Integer(        "intermediate_dimension",   (100, 500), log=True, default=500),
        Categorical(    "intermediate_activation",  ["relu", "selu", "tanh", "leakyrelu"], default="selu"),
        Integer(        "latent_dimension",         (10, 100), log=False, default=100),
        Categorical(    "solver",                   ["nadam"], default="nadam"),
        Float(          "learning_rate",            (1e-4, 1e-2), log=True, default=1e-3)
    ]
    configuration_space.add_hyperparameters(hyperparameters)
    forbidden_clauses = [
        ForbiddenGreaterThanRelation(configuration_space["latent_dimension"], configuration_space["intermediate_dimension"])
    ]
    configuration_space.add_forbidden_clauses(forbidden_clauses)
    if verbosity > 0: 
        print(f"Configuration space defined with estimated {configuration_space.estimate_size()} possible combinations.\n")


    if "torch" in framework:
        device = search_device(verbosity=verbosity)
        fia_vae_hptune = FIA_VAE_hptune( X, test_size=0.2, configuration_space=configuration_space, model_builder=FIA_VAE,
                                         device=device, workers=0, batch_size=64, verbosity=verbosity )
        
    elif "keras" in framework:
        fia_vae_hptune = FIA_VAE_hptune( X, test_size=0.2, configuration_space=configuration_space, model_builder=FIA_VAE,
                                         batch_size=64, verbosity=verbosity )
    
    else:
        raise(ValueError(f"The framework '{framework}' is not implemented. The framework must contain one of ['torch', 'keras']."))


    scenario = Scenario( fia_vae_hptune.configuration_space, deterministic=True,
                         n_trials=100000, min_budget=2, max_budget=100,
                         n_workers=1, output_directory=outdir,
                         walltime_limit=np.inf, cputime_limit=np.inf, trial_memory_limit=None )   # Max RAM in Bytes (not MB)
                        
    initial_design = MultiFidelityFacade.get_initial_design(scenario, n_configs=10)
    intensifier = Hyperband(scenario, incumbent_selection="highest_budget")
    facade = MultiFidelityFacade( scenario, fia_vae_hptune.train, 
                                  initial_design=initial_design, intensifier=intensifier,
                                  overwrite=overwrite, logging_level=30-verbosity*10 )
    time_step(message=f"SMAC defined. Overwriting: {overwrite}", verbosity=verbosity)


    if test_configuration:
        config = ConfigurationSpace(
                    {'input_dropout': 0.25, 'intermediate_activation': 'leakyrelu', 'intermediate_dimension': 200,
                    'intermediate_layers': 4, 'latent_dimension': 10, 'learning_rate': 0.001,
                    'original_dim': 825000, 'solver': 'nadam'}
                )
        test_train(smac_model=fia_vae_hptune, config=config, verbosity=verbosity)

    else:
        incumbent = run_optimization(facade=facade, smac_model=fia_vae_hptune, verbose_steps=10, verbosity=verbosity)

        save_runhistory(incumbent=incumbent, fascade=facade, run_dir=run_dir, verbosity=verbosity)



def time_step(message:str, verbosity:int=0):
    """
    Saves the time difference between last and current step
    """
    global last_timestamp
    global step
    global runtimes
    runtimes[f"{step}: {message}"] = [time.time() - last_timestamp]
    last_timestamp = time.time()
    step += 1
    if verbosity > 0: 
        print(message)


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
    time_step(message="Data loaded", verbosity=verbosity)
    return X


def search_device(verbosity:int=0):
    """
    Searches the fastest device for computation in pytorch
    Args:
        verbosity (int): level of verbosity
    Returns:
        device (str)
    """
    device = ( "cuda" if torch.cuda.is_available()
                    else "mps" if torch.backends.mps.is_available()
                    else "cpu" )
    if verbosity > 0:
        print(f"Using {device} device")
    return device


def test_train(smac_model, config:Configuration, verbosity:int=0):
    """
    Run a test training run of a model
    
    Args:
        smac_model: smac model to be used
        config (ConfigSpace.Configuration): configuration of test_run
        verbosity (int): verbosity of output
    """
    if verbosity > 0: 
        print("Test run:")
        print(config.get_default_configuration())
    smac_model.train(config.get_default_configuration(), seed=42, budget=2)


def ask_tell_optimization(facade, smac_model, n:int=10, verbosity:int=0):
    """
    Run the training run for n steps in a more verbose mode

    Args:
        facade: Facade used by smac
        smac_model: smac model that is used
        n (int): number of verbose runs
        verbosity (int): verbosity of output
    """
    for i in tqdm(range(n)):
        acc_time = time.time()
        info = facade.ask()
        assert info.seed is not None
        if verbosity > 1:
            print(f"Configuration: {dict(info.config)}")
        loss = smac_model.train(info.config, seed=info.seed, budget=info.budget)
        value = TrialValue(cost=loss, time=time.time()-acc_time, starttime=acc_time, endtime=time.time())

        facade.tell(info, value)


def run_optimization(facade, smac_model, verbose_steps:int=10, verbosity:int=0):
    """
    Perform optimization run with smac facade.

    Args:
        facade: SMAC facade
        smac_model: Model to supply for training
        verbose_steps (int): number of steps to be returned in a more verbose fashion
        verbosity (int): level of verbosity
    Returns:
        incumbent: best hyperparameter cominations
    """
    if verbosity > 0:
        print("Starting search:")
        ask_tell_optimization(facade=facade, smac_model=smac_model, n=verbose_steps, verbosity=verbosity)
    incumbent = facade.optimize()
    time_step(message="Search completed", verbosity=verbosity)
    return incumbent


def save_runhistory(incumbent, fascade, run_dir:str, verbosity:int=0):
    """
    Saves the history of one run

    Args:
        incumbent: The calculated incumbent (best hyperparameters)
        fascade: The fascade used for computation
        run_dir (str): The directory where the results are saved to
        verbosity (int): level of verbosity
    """
    best_hp = incumbent[0] if isinstance(incumbent, list) else incumbent
    if verbosity > 0: 
        print(f"The final incumbent cost is as: {fascade.validate(best_hp)}")

    results = pd.DataFrame(columns=["config_id", "config", "instance", "budget", "seed", "loss", "time", "status", "additional_info"])
    for trial_info, trial_value in fascade.runhistory.items():
        results.loc[len(results.index)] = [trial_info.config_id, dict(fascade.runhistory.get_config(trial_info.config_id)), trial_info.instance,
                                        trial_info.budget, trial_info.seed,
                                        trial_value.cost, trial_value.time, trial_value.status, trial_value.additional_info]
    results.to_csv(os.path.join(run_dir, "results_hp_search.tsv"), sep="\t")
    time_step(message="Saved history", verbosity=verbosity)

    # Runtime notation
    global runtimes
    runtimes["total"] = [np.sum(runtimes.values())]
    runtime_df = pd.DataFrame(runtimes)
    runtime_df.to_csv(os.path.join(run_dir, "runtimes.tsv"), sep="\t")
    if verbosity > 0: 
        print("Finished!")
    
    return (results, runtime_df)


if __name__ == "__main__":
    main()
