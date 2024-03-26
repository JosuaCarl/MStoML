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
from VAE import *

# Argument parser
parser = argparse.ArgumentParser(prog='VAE_smac_run',
                                description='Hyperparameter tuning for Variational Autoencoder with SMAC')
parser.add_argument('-d', '--data_dir')
parser.add_argument('-r', '--run_dir')
args = parser.parse_args()


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

    # Data Read-in
    binned_dfs = pd.read_csv(os.path.join(data_dir, "data_matrix.tsv"), sep="\t", index_col="mz", engine="pyarrow")
    binned_dfs[:] =  total_ion_count_normalization(binned_dfs)      # type: ignore

    X = binned_dfs.transpose()

    runtimes["data preparation"] = [time.time() - step_time]
    step_time = time.time()

    # Configuration space
    configuration_space = ConfigurationSpace()

    original_dim                = Constant('original_dim', X.shape[1])
    intermediate_neurons        = Integer('intermediate_neurons', (100, 10000), log=True, default=1000)
    intermediate_activation     = Categorical("intermediate_activation", ["relu", "tanh", "leakyrelu"], default="relu")
    input_dropout               = Float('input_dropout', (0.2, 0.8), default=0.5)
    intermediate_dropout        = Float('intermediate_dropout', (0.2, 0.8), default=0.5)
    latent_dimensions           = Integer('latent_dimensions', (10, 1000), log=False, default=100)
    kl_loss_scaler              = Float('kl_loss_scaler', (1e-3, 1e1), log=True, default=1e-2)
    solver                      = Categorical("solver", ["nadam"], default="nadam")
    learning_rate               = Float('learning_rate', (1e-4, 1e-2), log=True, default=1e-3)

    hyperparameters = [original_dim, intermediate_neurons, intermediate_activation, input_dropout, intermediate_dropout,
                    latent_dimensions, kl_loss_scaler, solver, learning_rate]
    configuration_space.add_hyperparameters(hyperparameters)


    outdir = Path(os.path.normpath(os.path.join(run_dir, "smac_vae")))
    fia_vae_hptune = FIA_VAE_hptune(X, test_size=0.2, configuration_space=configuration_space, model_builder=build_vae_ht_model, model_args={"classes": 1})

    # Define our environment variables
    scenario = Scenario( fia_vae_hptune.configuration_space, n_trials=10000,
                        deterministic=True,
                        min_budget=5, max_budget=100,
                        n_workers=1, output_directory=outdir,
                        walltime_limit=12*60*60, cputime_limit=np.inf, trial_memory_limit=int(2e12)    # Max RAM in Bytes (not MB)
                        )

    initial_design = MultiFidelityFacade.get_initial_design(scenario, n_configs=100)

    intensifier = Hyperband(scenario, incumbent_selection="highest_budget")

    # Create our SMAC object and pass the scenario and the train method
    smac = MultiFidelityFacade( scenario, fia_vae_hptune.train, 
                            initial_design=initial_design, intensifier=intensifier,
                            overwrite=False, logging_level=20
                            )
    
    runtimes["SMAC definition"] = [time.time() - step_time]
    step_time = time.time()
    
    # Optimization run
    incumbent = smac.optimize()

    runtimes["SMAC optimization"] = [time.time() - step_time]
    step_time = time.time()

    # Saving incumbent
    if isinstance(incumbent, list):
        best_hp = incumbent[0]
    else: 
        best_hp = incumbent
    incumbent_cost = smac.validate(best_hp)

    results = pd.DataFrame(columns=["config_id", "config", "instance", "budget", "seed", "loss", "time", "status", "additional_info"])
    for trial_info, trial_value in smac.runhistory.items():
        results.loc[len(results.index)] = [trial_info.config_id, dict(smac.runhistory.get_config(1)), trial_info.instance,
                                        trial_info.budget, trial_info.seed,
                                        trial_value.cost, trial_value.time, trial_value.status, trial_value.additional_info]
    results.to_csv(os.path.join(run_dir, "results_hp_search.tsv"), sep="\t")

    runtimes["Saving results"] = [time.time() - step_time]
    step_time = time.time()

    # Runtime
    total_runtime = time.time() - start
    runtimes["total"] = [total_runtime]
    runtimes = pd.DataFrame(runtimes)
    runtimes.index = ["total", "per sample", "per file"]    # type: ignore
    runtimes.to_csv(os.path.join(run_dir, "runtimes.tsv"), sep="\t")

__main__()
