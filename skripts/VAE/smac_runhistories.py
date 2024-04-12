import sys
import os

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
    parser.add_argument('-i', '--in_dirs')
    parser.add_argument('-o', '--out_dir')
    parser.add_argument('-v', '--verbosity')
    parser.add_argument('-f', '--framework')
    args = parser.parse_args()

    in_dirs, out_dir = [os.path.normpath(os.path.join(os.getcwd(), d)) for d in  [args.in_dirs.split(","), args.out_dir]]
    verbosity =  int(args.verbosity) if args.verbosity else 0
    framework = args.framework
    smac_dirs = [Path(os.path.normpath(os.path.join(in_dir, f"smac_vae_{framework}"))) for in_dir in in_dirs]
    time_step(message="Setup loaded", verbosity=verbosity)

    runhistories = read_runhistories(folder_paths=smac_dirs, verbosity=verbosity)

    save_runhistory(runhistory=runhistories, out_dir=out_dir)

    if verbosity > 0:
        print("Saved Runhistories.")


def read_runhistories(folder_paths:list, skip_dirs:list, verbosity:int=0) -> RunHistory:
    """
    Read in different runhistories

    Args:
        folder_paths (str): Path to the folders with two subdirectories, containing 'configspace.json'
    Returns:
        runhistories (RunHistory)
    """
    runhistories = RunHistory()
    for folder_path in folder_paths:
        for dir in os.listdir(folder_path):
            if dir in skip_dirs:
                continue
            for sub in os.listdir(os.path.join(folder_path, dir)):
                with open(os.path.join(folder_path, dir, sub, 'configspace.json'), 'r') as f:
                    json_string = f.read()
                    configuration_space = cs_json.read(json_string)
                runhistory = RunHistory()
                runhistory.load(filename=os.path.join(folder_path, dir, sub, 'runhistory.json'), configspace=configuration_space)
                runhistories.update(runhistory)
    
    if verbosity > 0:
        print(f"Best out of {len(runhistories)} run histories:")
        best_config = runhistories.get_configs(sort_by="cost")[0]
        print(best_config)
        print(f"With minimal cost: {runhistories.get_min_cost(best_config)}")

    return runhistories


def save_runhistory(runhistory, out_dir:str, verbosity:int=0) -> RunHistory:
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
    results.to_csv(os.path.join(out_dir, "results_hp_search.tsv"), sep="\t")

    return results


if __name__ == "__main__":
    main()