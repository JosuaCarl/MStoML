# Imports
import os
import time
import sys
import argparse

# Append paths to scripting files
sys.path.append( '../FIA' )

# import methods from FIA python script
from FIA import *

# Argument parser
parser = argparse.ArgumentParser(prog='FIA_oms',
                                description='Flow-injection analysis with OpenMS python bindings.')
parser.add_argument('-d', '--data_dir')      # option that takes a value
parser.add_argument('-r', '--run_dir')
args = parser.parse_args()
tqdm.pandas()

def __main__():
    # Runtime
    runtimes = {}
    start = time.time()

    # set path to files and workfolder
    data_dir, run_dir = (args.data_dir, args.run_dir)
    data_dir = os.path.normpath(os.path.join(os.getcwd(), data_dir))
    run_dir = os.path.normpath(os.path.join(os.getcwd(), run_dir))

    runtimes["setup"] = [time.time() - start]
    step_time = time.time()

    # Load Data
    fia_df = load_fia_df(data_dir, ".mzML")
    runtimes["trimming"] = [time.time() - step_time]
    step_time = time.time()

    # Binning
    # Computes mean, media or sum of binned peaks (median needs ~2* more time)
    print("Binning experiments:")
    fia_df["experiment"] = fia_df["experiment"].progress_apply(lambda experiment: limit_experiment(experiment, 51, 1699, 2*10**6, statistic="sum", deepcopy=True))
    runtimes["binning"] = [time.time() - step_time]
    step_time = time.time()

    # Summing spectra
    print("Summing spectra:")
    fia_df["sum_spectra"] = fia_df["experiment"].progress_apply(lambda experiment: sum_spectra(experiment)) # type: ignore
    runtimes["summing spectra"] = [time.time() - step_time]
    step_time = time.time()
    

    # Equal sample and polarity merging
    print("Sample merging:")
    merge_dict = make_merge_dict(data_dir, file_ending=".mzML")
    comb_df = pd.DataFrame(columns=["polarity", "sample", "comb_experiment"])
    for sample in tqdm(fia_df["sample"].unique()):
        for polarity in fia_df["polarity"].unique():    
            uniq_samples = fia_df["polarity"] == polarity and fia_df["sample"] == sample
            comb_df.loc[len(comb_df.index)] = [polarity, sample, combine_spectra_experiments(fia_df.loc[uniq_samples]["sum_spectra"].to_list())]
    runtimes["sample merging"] = [time.time() - step_time]
    step_time = time.time()


    # Clustering
    print("Clustering:")
    comb_df["clustered_experiment"] = comb_df["comb_experiment"].progress_apply(lambda experiment: cluster_sliding_window(experiment, window_len=2000, window_shift=1000, threshold=0.07**2))
    runtimes["clustering"] = [time.time() - step_time]
    step_time = time.time()


    # Merging Polarities
    print("Polarity merging:")
    for sample in tqdm(fia_df["sample"].unique()):
        uniq_samples = fia_df["polarity"] == polarity and fia_df["sample"] == sample
        all_merged_df = merge_mz_tolerance(comb_df, charge=1, tolerance=1e-3)
        all_merged_df.to_csv(os.path.join(run_dir, f"{sample}.tsv"), sep="\t", index=False)
        runtimes["merging"] = [time.time() - step_time]
        step_time = time.time()


    # Runtime
    runtime = time.time() - start
    runtimes["total"] = [runtime]
    runtimes = pd.DataFrame(runtimes)
    runtimes.loc[1] = np.array(runtimes.loc[runtimes.index[0]].values) / len(merge_dict.keys())
    runtimes.loc[2] = np.array(runtimes.loc[runtimes.index[0]].values) / len(sum(merge_dict.values(), []))
    runtimes.index = ["total", "per sample", "per file"]    # type: ignore
    runtimes.to_csv(os.path.join(run_dir, "runtimes.tsv"), sep="\t")
    print(f"approx. runtime: {int(runtime)} s")

__main__()
