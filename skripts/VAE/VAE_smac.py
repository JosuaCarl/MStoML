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

from sklearn.model_selection import train_test_split

from ConfigSpace import Categorical, Configuration, ConfigurationSpace, Float, Integer, Constant, ForbiddenGreaterThanRelation
from smac import MultiFidelityFacade
from smac import Scenario
from smac.intensifier.hyperband import Hyperband
from smac.runhistory.dataclasses import TrialValue

# Logging (time and steps)
last_timestamp = time.time()
step = 0
runtimes = {}


def main(args):
    """
    Hyperparameter optimization with SMAC3
    """
    data_dir, run_dir = [os.path.normpath(os.path.join(os.getcwd(), d)) for d in  [args.data_dir, args.run_dir]]
    overwrite =  bool(args.overwrite)
    backend_name = args.backend
    computation = args.computation
    gpu = computation == "gpu"
    name = args.name if args.name else None
    project = f"smac_vae_{backend_name}_{computation}_{name}" if name else f"smac_vae_{backend_name}_{computation}"
    verbosity =  args.verbosity if args.verbosity else 0
    outdir = Path(os.path.normpath(os.path.join(run_dir, project)))

    if verbosity > 0:
        print(f"Using {keras.backend.backend()} as backend")
    time_step(message="Setup loaded", verbosity=verbosity, min_verbosity=1)

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


    fia_vae_hptune = FIA_VAE_tune( X, test_size=0.2, configuration_space=configuration_space, model_builder=FIA_VAE,
                                   batch_size=64, log_dir=os.path.join(outdir, "log"), verbosity=verbosity, gpu=gpu)


    scenario = Scenario( fia_vae_hptune.configuration_space, deterministic=True,
                         n_trials=100000, min_budget=2, max_budget=100,
                         n_workers=1, output_directory=outdir,
                         walltime_limit=np.inf, cputime_limit=np.inf, trial_memory_limit=None )   # Max RAM in Bytes (not MB)
                        
    initial_design = MultiFidelityFacade.get_initial_design(scenario, n_configs=10)
    intensifier = Hyperband(scenario, incumbent_selection="highest_budget")
    facade = MultiFidelityFacade( scenario, fia_vae_hptune.train, 
                                  initial_design=initial_design, intensifier=intensifier,
                                  overwrite=overwrite, logging_level=30-verbosity*10 )
    time_step(message=f"SMAC defined. Overwriting: {overwrite}", verbosity=verbosity, min_verbosity=1)

    incumbent = run_optimization(facade=facade, smac_model=fia_vae_hptune, verbose_steps=10, verbosity=verbosity)

    best_hp = validate_incumbent(incumbent=incumbent, fascade=facade, verbosity=verbosity)

    save_runtime(run_dir=run_dir, verbosity=verbosity)


# METHODS
def time_step(message:str, verbosity:int=0, min_verbosity:int=1):
    """
    Saves the time difference between last and current step

    Args:
        message (str): Message, that will be printed and saved, along with runtime.
        verbosity (int): Current verbosity. 
        min_verbosity (int): If verbosity >= min_verbosity, print message.
    """
    global last_timestamp
    global step
    global runtimes
    runtimes[f"{step}: {message}"] = time.time() - last_timestamp
    if verbosity >= min_verbosity: 
        print(f"{message} ({runtimes[f'{step}: {message}']}s)")
    last_timestamp = time.time()
    step += 1


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
        if verbosity >= 2:
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
    if verbosity >= 1:
        print("Starting search:")
        ask_tell_optimization(facade=facade, smac_model=smac_model, n=verbose_steps, verbosity=verbosity)
    incumbent = facade.optimize()
    time_step(message="Search completed", verbosity=verbosity)
    return incumbent


def validate_incumbent(incumbent, fascade, run_dir:str, verbosity:int=0):
    """
    Saves the history of one run

    Args:
        incumbent: The calculated incumbent (best hyperparameters)
        fascade: The fascade used for computation
        verbosity (int): level of verbosity
    """
    best_hp = incumbent[0] if isinstance(incumbent, list) else incumbent
    return best_hp 


def save_runtime(run_dir, verbosity:int=0):
    global runtimes
    runtimes["total"] = [np.sum(runtimes.values())]
    runtime_df = pd.DataFrame(runtimes)
    runtime_df.to_csv(os.path.join(run_dir, "runtimes.tsv"), sep="\t")
    if verbosity >= 1: 
        print("Finished!")
    
    return runtime_df


class FIA_VAE_tune:
    """
    Class for running the SMAC3 tuning
    """
    def __init__(self, X, test_size:float, configuration_space:ConfigurationSpace, model_builder,
                 log_dir:str, batch_size:int=16, verbosity:int=0, gpu:bool=False):
        self.configuration_space = configuration_space
        self.model_builder = model_builder
        self.training_data, self.test_data = train_test_split(X, test_size=test_size)
        self.batch_size = batch_size
        self.log_dir = log_dir
        self.verbosity = verbosity
        self.gpu = gpu

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
        time_step("Start", verbosity=self.verbosity, min_verbosity=2)
        keras.utils.set_random_seed(seed)

        # Definition
        model = self.model_builder(config)
        if self.verbosity >= 3:
            model.vae.summary()
            print_utilization(gpu=self.gpu)
        time_step("Model built", verbosity=self.verbosity, min_verbosity=2)

        # Fitting
        callbacks = []
        if self.verbosity >= 2:
            log_dir = os.path.join(self.log_dir,  datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            callbacks.append( TensorBoard(log_dir=log_dir, write_graph=True, write_images=True, update_freq='epoch') )

        model.fit(x=self.training_data, y=self.training_data, validation_split=0.2,
                  batch_size=self.batch_size, epochs=int(budget),
                  callbacks=callbacks, verbose=self.verbosity)

        if self.verbosity >= 3:
            print("After training utilization:")
            print_utilization(gpu=self.gpu)
        time_step("Model trained", verbosity=self.verbosity, min_verbosity=2)

        # Evaluation
        loss, recon_loss, kl_loss = model.evaluate(self.test_data, self.test_data,
                                                   batch_size=self.batch_size, verbose=self.verbosity)
        time_step("Model evaluated", verbosity=self.verbosity, min_verbosity=2)
        
        # Clearing model parameters
        keras.backend.clear_session()
        time_step("Session cleared", verbosity=self.verbosity, min_verbosity=2)
                
        return loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='VAE_smac_run',
                                description='Hyperparameter tuning for Variational Autoencoder with SMAC')
    parser.add_argument('-d', '--data_dir', required=True)
    parser.add_argument('-r', '--run_dir', required=True)
    parser.add_argument('-o', '--overwrite', action="store_true", required=False)
    parser.add_argument('-b', '--backend',  required=True)
    parser.add_argument('-c', '--computation', required=True)
    parser.add_argument('-n', '--name', required=False)
    parser.add_argument('-v', '--verbosity', type=int, required=True)
    args = parser.parse_args()

    os.environ["KERAS_BACKEND"] = args.backend

    # Doesn't work with Slurm, because __file__ variable is not preserved when copying to cluster
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    sys.path.append( os.path.normpath(os.path.join( dir_path, '..' )))
    """
    sys.path.append("..")
    from helpers.pc_stats import *
    from VAE.vae import keras, np, datetime, read_data, TensorBoard, FIA_VAE

    main(args=args)