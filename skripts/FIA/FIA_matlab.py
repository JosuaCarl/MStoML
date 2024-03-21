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
    fia_df = load_fia_df(data_dir, ".mzML", data_load=False)
    runtimes["trimming"] = [time.time() - step_time]
    step_time = time.time()
    
    for sample in tqdm(fia_df["sample"].unique()):
        # For multi-step processing
        if os.path.isfile(os.path.join(run_dir, f"{sample}.tsv")):
            continue

        fia_df_tmp = fia_df.loc[fia_df["sample"]==sample,:]
        fia_df_tmp.loc[:,"experiment"] = fia_df_tmp["experiment"].apply(lambda experiment: load_experiment(experiment))
        
        # Binning
        # Computes mean, media or sum of binned peaks (median needs ~2* more time)
        print("Binning experiments:")
        fia_df_tmp.loc[:,"experiment"] = fia_df_tmp["experiment"].apply(lambda experiment: limit_experiment(experiment, 51, 1699, 2*10**6, statistic="sum", deepcopy=True))

        # Summing spectra
        print("Summing spectra:")
        fia_df_tmp.loc[:,"sum_spectra"] = fia_df_tmp["experiment"].apply(lambda experiment: sum_spectra(experiment)) # type: ignore

        # Equal sample and polarity merging
        print("Sample merging:")
        merge_dict = make_merge_dict(data_dir, file_ending=".mzML")
        comb_df = pd.DataFrame(columns=["polarity", "sample", "comb_experiment"])
        for polarity in fia_df_tmp["polarity"].unique():    
            uniq_samples = (fia_df_tmp["polarity"] == polarity) & (fia_df_tmp["sample"] == sample)
            comb_df.loc[len(comb_df.index)] = [polarity, sample, combine_spectra_experiments(fia_df_tmp.loc[uniq_samples]["sum_spectra"].to_list())]


        # Clustering
        print("Clustering:")
        comb_df.loc[:,"clustered_experiment"] = comb_df["comb_experiment"].apply(lambda experiment: cluster_sliding_window(experiment, window_len=2000, window_shift=1000, threshold=0.07**2))


        # Merging Polarities
        print("Polarity merging:")
        for sample in fia_df_tmp["sample"].unique():
            uniq_samples = (fia_df_tmp["polarity"] == polarity) & (fia_df_tmp["sample"] == sample)
            all_merged_df = merge_mz_tolerance(comb_df, charge=1, tolerance=1e-3, binned=True)
            all_merged_df.to_csv(os.path.join(run_dir, f"{sample}.tsv"), sep="\t", index=False)


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
