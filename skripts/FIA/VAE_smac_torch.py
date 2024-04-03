#!/usr/bin/env python3
#SBATCH --job-name VAE_tuning
#SBATCH --time 12:00:00
#SBATCH --mem 450G
#SBATCH --partition cpu2-hm
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 1

# processors: cpu1, cpu2-hm, gpu-a30
# imports
import sys
import time
import argparse

sys.path.append( '../FIA' )
sys.path.append( '../ML' )
sys.path.append( '..' )

from FIA import *
from ML4com import *
from helpers import *
from VAE_torch import *

# Argument parser
parser = argparse.ArgumentParser(prog='VAE_smac_run',
                                description='Hyperparameter tuning for Variational Autoencoder with SMAC')
parser.add_argument('-d', '--data_dir')
parser.add_argument('-r', '--run_dir')
args = parser.parse_args()

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


def __main__():
    # Runtime logging
    runtimes = {}
    start = time.time()

    print([dir for dir in  [args.data_dir, args.run_dir]])
    print(os.getcwd())
    # set path to files and workfolder
    data_dir, run_dir = [os.path.normpath(os.path.join(os.getcwd(), d)) for d in  [args.data_dir, args.run_dir]]

    runtimes["setup"] = [time.time() - start]
    step_time = time.time()
    print("Setup loaded.")

    # Data Read-in
    binned_dfs = pd.read_csv(os.path.join(data_dir, "data_matrix.tsv"), sep="\t", index_col="mz", engine="pyarrow")
    binned_dfs[:] =  total_ion_count_normalization(binned_dfs)      # type: ignore

    X = binned_dfs.transpose()

    runtimes["data preparation"] = [time.time() - step_time]
    step_time = time.time()
    print("Data loaded.")

    configuration_space = ConfigurationSpace(seed=42)
    original_dim                = Constant('original_dim', X.shape[1])
    input_dropout               = Float('input_dropout', (0.0, 0.5), default=0.25)
    intermediate_layers         = Integer('intermediate_layers', (1, 3), default=2)
    intermediate_dimension      = Integer('intermediate_dimension', (500, 2000), log=True, default=2000)
    intermediate_activation     = Categorical("intermediate_activation", ["relu", "selu","tanh", "leakyrelu"], default="selu")
    latent_dimension            = Integer('latent_dimension', (100, 1000), log=False, default=1000)
    solver                      = Categorical("solver", ["nadam"], default="nadam")
    learning_rate               = Float('learning_rate', (1e-4, 1e-2), log=True, default=1e-3)

    hyperparameters = [original_dim, input_dropout,
                       intermediate_layers, intermediate_dimension, intermediate_activation,
                       latent_dimension, solver, learning_rate]
    configuration_space.add_hyperparameters(hyperparameters)

    latent_limiter = ForbiddenGreaterThanRelation(configuration_space["latent_dimension"], configuration_space["intermediate_dimension"])
    configuration_space.add_forbidden_clauses([latent_limiter])

    print(f"Configuration space defined with estimated {configuration_space.estimate_size()} possible combinations.\n")

    outdir = Path(os.path.normpath(os.path.join(run_dir, "smac_vae")))
    fia_vae_hptune = FIA_VAE_hptune(X, test_size=0.2, configuration_space=configuration_space, model_builder=FIA_VAE,
                                    device=device, workers=1, batch_size=64, verbose=False)

    # Define our environment variables
    scenario = Scenario( fia_vae_hptune.configuration_space, n_trials=1000,
                        deterministic=True,
                        min_budget=5, max_budget=100,
                        n_workers=1, output_directory=outdir,
                        walltime_limit=12*60*60, cputime_limit=np.inf, trial_memory_limit=None    # Max RAM in Bytes (not MB)
                        )

    initial_design = MultiFidelityFacade.get_initial_design(scenario, n_configs=10)

    intensifier = Hyperband(scenario, incumbent_selection="highest_budget")

    # Create our SMAC object and pass the scenario and the train method
    smac = MultiFidelityFacade( scenario, fia_vae_hptune.train, 
                            initial_design=initial_design, intensifier=intensifier,
                            overwrite=True, logging_level=20
                            )
    
    runtimes["SMAC definition"] = [time.time() - step_time]
    step_time = time.time()
    print("SMAC defined.\n")
    
    # Optimization run
    print("Starting search:")
    for i in tqdm(range(100)):                         # Test first 5 runs to see if process works
        acc_time = time.time()
        info = smac.ask()
        assert info.seed is not None

        cost = fia_vae_hptune.train(info.config, seed=info.seed, budget=info.budget)
        value = TrialValue(cost=cost, time=time.time()-acc_time, starttime=acc_time, endtime=time.time())

        smac.tell(info, value)


    incumbent = smac.optimize()

    runtimes["SMAC optimization"] = [time.time() - step_time]
    step_time = time.time()
    print("Search completed.")

    # Saving incumbent
    if isinstance(incumbent, list):
        best_hp = incumbent[0]
    else: 
        best_hp = incumbent
    print(f"The final incumbent cost is as: {smac.validate(best_hp)}")

    results = pd.DataFrame(columns=["config_id", "config", "instance", "budget", "seed", "loss", "time", "status", "additional_info"])
    for trial_info, trial_value in smac.runhistory.items():
        results.loc[len(results.index)] = [trial_info.config_id, dict(smac.runhistory.get_config(1)), trial_info.instance,
                                        trial_info.budget, trial_info.seed,
                                        trial_value.cost, trial_value.time, trial_value.status, trial_value.additional_info]
    results.to_csv(os.path.join(run_dir, "results_hp_search.tsv"), sep="\t")

    runtimes["Saving results"] = [time.time() - step_time]
    step_time = time.time()
    print("Results saved.")

    # Runtime
    total_runtime = time.time() - start
    runtimes["total"] = [total_runtime]
    runtimes = pd.DataFrame(runtimes)
    runtimes.index = ["total", "per sample", "per file"]    # type: ignore
    runtimes.to_csv(os.path.join(run_dir, "runtimes.tsv"), sep="\t")
    print("Finished!")

__main__()
