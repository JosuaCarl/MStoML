#!/usr/bin/env python3
#SBATCH --job-name Runhistory-collection
#SBATCH --time 00:30:00
#SBATCH --nodes 2
#SBATCH --partition cpu1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 1

import sys
import os
import json

from smac import RunHistory
from ConfigSpace.read_and_write import json as cs_json

import matplotlib.pyplot as plt

sys.path.append( '..' )
from VAE.VAE_smac import *
from helpers.pc_stats import *


def main():
    """
    Interpretation of SMAC run
    """
    parser = argparse.ArgumentParser(prog='Interpret_smac',
                                     description='Hyperparameter tuning for Variational Autoencoder with SMAC')
    parser.add_argument('-i', '--in_dir', required=True)
    parser.add_argument('-o', '--out_dir', required=True)
    parser.add_argument('-b', '--backend',  required=True)
    parser.add_argument('-c', '--computation', required=True)
    parser.add_argument('-n', '--name', required=False)
    parser.add_argument('-v', '--verbosity', required=True)
    args = parser.parse_args()

    in_dir = os.path.normpath(os.path.join(os.getcwd(), args.in_dir))
    out_dir = os.path.normpath(os.path.join(os.getcwd(), args.out_dir))
    verbosity =  int(args.verbosity) if args.verbosity else 0
    backend_name = args.backend
    computation = args.computation
    name = args.name if args.name else None
    project = f"smac_vae_{backend_name}_{computation}_{name}" if name else f"smac_vae_{backend_name}_{computation}"
    smac_dir = Path(os.path.normpath(os.path.join(in_dir, project)))
    time_step(message="Setup loaded", verbosity=verbosity, min_verbosity=1)

    best_config, runhistories = read_runhistories(folder_path=smac_dir, skip_dirs="mlruns", verbosity=verbosity)

    with open(os.path.join(in_dir, project, f"best_config.json"), "w") as outfile:
        json.dump(dict(best_config), outfile)
    
    save_runhistory(runhistory=runhistories, project=project, out_dir=out_dir)

    if verbosity >= 1:
        print("Saved Runhistories.")



def read_runhistories(folder_path, skip_dirs:list, verbosity:int=0) -> RunHistory:
    """
    Read in different runhistories

    Args:
        folder_path (str): Path to the folders with two subdirectories, containing 'configspace.json'
    Returns:
        runhistories (RunHistory)
    """
    runhistories = RunHistory()
    for d in os.listdir(folder_path):
        if d in skip_dirs or not os.path.isdir(d):
            continue
        for sub in tqdm(os.listdir(os.path.join(folder_path, d))):
            with open(os.path.join(folder_path, d, sub, 'configspace.json'), 'r') as f:
                json_string = f.read()
                configuration_space = cs_json.read(json_string)
            runhistory = RunHistory()
            runhistory.load(filename=os.path.join(folder_path, d, sub, 'runhistory.json'), configspace=configuration_space)
            runhistories.update(runhistory)
    
    if verbosity > 0:
        print(f"Best out of {len(runhistories)} run histories:")
        best_config = runhistories.get_configs(sort_by="cost")[0]
        print(best_config)
        print(f"With minimal cost: {runhistories.get_min_cost(best_config)}")

    return (best_config, runhistories)


def save_runhistory(runhistory, project:str, out_dir:str, verbosity:int=0) -> RunHistory:
    """
    Saves the history of one run

    Args:
        runhistory (smac.RunHistory): Runhistory of one or multiple run(s)
        run_dir (str): The directory where the results are saved to
        verbosity (int): level of verbosity
    Returns:
        results (pd.DataFrame): Summary of the runhistory in DataFrame format
    """
    results = pd.DataFrame(columns=["config_id", "config", "instance", "budget", "seed", "loss", "time", "status", "additional_info"])
    for trial_info, trial_value in runhistory.items():
        results.loc[len(results.index)] = [trial_info.config_id, dict(runhistory.get_config(trial_info.config_id)), trial_info.instance,
                                        trial_info.budget, trial_info.seed,
                                        trial_value.cost, trial_value.time, trial_value.status, trial_value.additional_info]
    results.to_csv(os.path.join(out_dir, f"results_hp_search{project}.tsv"), sep="\t")

    return results


if __name__ == "__main__":
    main()