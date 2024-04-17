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

# Doesn't work with Slurm, because __file__ variable is not copied
"""
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append( os.path.normcase(os.path.join( dir_path, '..' )))
print(os.path.normpath(os.path.join( dir_path, '..' )))
"""
sys.path.append("..")
from helpers.pc_stats import *
from FIA.FIA import *
from VAE.VAE import *

# Argument parser
parser = argparse.ArgumentParser(prog='VAE_smac_run',
                                description='Hyperparameter tuning for Variational Autoencoder with SMAC')
parser.add_argument('-d', '--data_dir')
parser.add_argument('-r', '--run_dir')
parser.add_argument('-v', '--verbosity')
parser.add_argument('-f', '--framework')
parser.add_argument('-o', '--overwrite')
args = parser.parse_args()


# Logging (time and steps)
last_timestamp = time.time()
step = 0
runtimes = {}

def main():
    """
    Hyperparameter optimization with SMAC3
    """
    data_dir, run_dir = [os.path.normpath(os.path.join(os.getcwd(), d)) for d in  [args.data_dir, args.run_dir]]
    overwrite, verbosity = ( bool(args.overwrite), int(args.verbosity) )
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
        fia_vae = FIA_VAE(framework)
        fia_vae_hptune = FIA_VAE_tune_torch( X, test_size=0.2, configuration_space=configuration_space, model_builder=FIA_VAE_torch,
                                             device=device, workers=0, batch_size=64, log_dir=os.path.join(outdir, "log"), verbosity=verbosity )
        
    elif "keras" in framework:
        fia_vae_hptune = FIA_VAE_tune_keras( X, test_size=0.2, configuration_space=configuration_space, model_builder=FIA_VAE_keras,
                                                   batch_size=64, log_dir=os.path.join(outdir, "log"), verbosity=verbosity )
    
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

    incumbent = run_optimization(facade=facade, smac_model=fia_vae_hptune, verbose_steps=10, verbosity=verbosity)

    best_hp = validate_incumbent(incumbent=incumbent, fascade=facade, verbosity=verbosity)

    save_runtime(run_dir=run_dir, verbosity=verbosity)


# METHODS
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
    if verbosity > 0: 
        print("Finished!")
    
    return runtime_df

# CLASSES
class FIA_VAE_tune_torch:
    def __init__(self, X, test_size:float, configuration_space:ConfigurationSpace, model_builder,
                 device:str, log_dir:str, workers:int=1, batch_size:int=16, verbosity:int=0):
        """
        X: Tensor
        test_size: The fraction as a float to be tested upon
        configuration_space: ConfigurationSpace from configspace (hyperparameters)
        device: device to be used for computation
        batch_size: Number of samples to be used at once as an integer
        verbosity: verbosity level as an integer [0: No output, 1: Summary output, >1: Model and usage output]
        """
        self.device                 = torch.device( device )
        self.configuration_space    = configuration_space
        self.model_builder          = model_builder
        training_data, test_data    = train_test_split(X, test_size=test_size)
        self.training_data          = torch.tensor(training_data.values).to(torch.float32).to( device )
        self.test_data              = torch.tensor(test_data.values).to(torch.float32).to ( device )
        self.workers                = workers
        self.batch_size             = batch_size
        self.writer                 = SummaryWriter(os.path.join(log_dir,  datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
        self.verbosity              = verbosity

    
    def seed_worker(self, worker_id):
        """
        Initalizes all modules with relevant random number generators to a seed to control randomness

        Args:
            worker_id: Used to be able to receive input from different workers
        """
        worker_seed = torch.initial_seed()
        random.seed(worker_seed)
        rng = np.random.default_rng(worker_seed)

    def train_epoch(self, model, data_loader, optimizer):
        """
        Train the model for one epoch

        Args:
            model
            data_loader
            optimizer
        """
        sum_loss = 0
        for data in data_loader:
            optimizer.zero_grad()  # Zero the gradients
            output = model(data)  # Forward pass
            loss = output.loss
            sum_loss += loss
            loss.backward()
            optimizer.step()  # Update the model parameters

        return sum_loss
    
    def evaluate(self, model, data_loader):
        """
        Evaluate the model

        Args:
            model
            data_loader
        
        Returns:
            Average loss of the model
        """
        sum_loss = 0
        for data in data_loader:
            with torch.no_grad():
                output = model(data) 
                sum_loss += output.loss
        return sum_loss / len(data_loader)
    
    def train(self, config: Configuration, seed:int=0, budget:int=10):
        """
        Method to train the model

        Args:
            config: Configuration to be trained upon
            seed: initializing seed
            budget: number of epochs to be used in training
        
        Returns:
            Average loss of the model
        """
        t = time.time()        
        generator = torch.Generator()
        generator.manual_seed(seed)

        # Dataset Loading
        train_loader = DataLoader(self.training_data, batch_size=self.batch_size,
                                  num_workers=self.workers, worker_init_fn=self.seed_worker,
                                  generator=generator, pin_memory=False )
        test_loader = DataLoader(self.test_data, batch_size=self.batch_size,
                                  num_workers=self.workers, worker_init_fn=self.seed_worker,
                                  generator=generator, pin_memory=False )
        
        # Definition
        model = self.model_builder(config).to( self.device )
        if self.verbosity > 1:
            if self.verbosity > 2:
                print(model)
                summary(model, inpute_size=self.training_data.shape, mode="train", device=self.device)
                print_utilization()
            print(f"Model built in {time.time()-t}s")
            t = time.time()
        
        # Fitting
        optimizer = model.get_solver( config["solver"] )(model.parameters(), lr=config["learning_rate"])
        model.init_weights()
        if self.verbosity > 2:
            for epoch in tqdm(range(int(budget))):
                loss = self.train_epoch(model=model, data_loader=train_loader, optimizer=optimizer)
                self.writer.add_scalar("Training loss", loss, epoch)
        else:
            for epoch in range(int(budget)):
                loss = self.train_epoch(model=model, data_loader=train_loader, optimizer=optimizer)
                self.writer.add_scalar("Training loss", loss, epoch)

        if self.verbosity > 1:
            if self.verbosity > 2:
                print("After training utilization:")
                print_utilization()
            print(f"Model trained in {time.time()-t}s")
            t = time.time()
        
        # Evaluation
        avg_loss = self.evaluate(model, test_loader)
        self.writer.add_scalar("Validation loss", avg_loss)
        self.writer.flush()
        if self.verbosity > 1:
            print(f"Model evaluated in {time.time()-t}s")

        return avg_loss.cpu()


class FIA_VAE_tune_keras:
    def __init__(self, X, test_size:float, configuration_space:ConfigurationSpace, model_builder,
                 log_dir:str, batch_size:int=16, verbosity:int=0):
        self.configuration_space = configuration_space
        self.model_builder = model_builder
        self.training_data, self.test_data = train_test_split(X, test_size=test_size)
        self.batch_size = batch_size
        self.log_dir = log_dir
        self.verbosity = verbosity

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
        t = time.time()
        keras.utils.set_random_seed(seed)

        # Definition
        model = self.model_builder(config)
        if self.verbosity > 1:
            if self.verbosity > 2:
                model.vae.summary()
                print_utilization()
            print(f"Model built in {time.time()-t}s")
            t = time.time()

        # Fitting
        model.train(self.training_data, self.training_data, epochs=int(budget),
                    batch_size=self.batch_size, log_dir=self.log_dir, verbosity=self.verbosity)
        if self.verbosity > 1:
            if self.verbosity > 2:
                print("After training utilization:")
                print_utilization()
            print(f"Model trained in {time.time()-t}s")
            t = time.time()

        # Evaluation
        loss, mse = model.evaluate(self.test_data, verbosity=self.verbosity)
        if self.verbosity > 1:
            print(f"Model evaluated in {time.time()-t}s")
        
        # Clearing model parameters
        keras.backend.clear_session()
        if self.verbosity > 1:
            print(f"Session cleared in ({time.time()-t}s)")
                
        return loss


if __name__ == "__main__":
    main()
