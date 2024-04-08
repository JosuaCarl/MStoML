# Imports
import os
import time
import sys
import argparse

# import methods from FIA python script
from skripts.FIA.fia import *

# Argument parser
parser = argparse.ArgumentParser(prog='FIA_oms',
                                description='Flow-injection analysis with OpenMS python bindings.')
parser.add_argument('-d', '--data_dir')      # option that takes a value
parser.add_argument('-r', '--run_dir')
args = parser.parse_args()

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


    # Trim values
    print("Trim values:")
    trim_dir = trim_threshold_batch(data_dir, run_dir, file_ending=".mzML", threshold=1e3, deepcopy=False)
    runtimes["trimming"] = [time.time() - step_time]
    step_time = time.time()


    # Centroiding
    print("Centroiding:")
    centroid_dir = centroid_batch(trim_dir, run_dir, file_ending=".mzML", instrument="TOF",
                                   signal_to_noise=2.0, spacing_difference=1.5,
                                   peak_width=0.0, sn_bin_count=100, nr_iterations=5, sn_win_len=20.0,
                                   check_width_internally="false", ms1_only="true", clear_meta_data="false",
                                   deepcopy=False
                                 )
    runtimes["centroiding"] = [time.time() - step_time]
    step_time = time.time()


    # Merging
    print("Merging:")
    merge_dir = merge_batch(centroid_dir, run_dir, file_ending=".mzML", method="block_method",
                            mz_binning_width=10.0, mz_binning_width_unit="ppm",
                            ms_levels=[1], sort_blocks="RT_ascending",
                            rt_block_size=None, rt_max_length=0.0,
                            )
    runtimes["merging"] = [time.time() - step_time]
    step_time = time.time()
    

    # Over-file merging
    print("Sample merging:")
    merge_dict = make_merge_dict(merge_dir, file_ending=".mzML")
    for sample, files in merge_dict.items():
        merged_exp = merge_experiments(files, run_dir, file_ending=".mzML", method="block_method",
                                        mz_binning_width=10.0, mz_binning_width_unit="ppm",
                                        ms_levels=[1], sort_blocks="RT_ascending",
                                        rt_block_size=None, rt_max_length=0.0,
                                        )
        merged_exp.setLoadedFilePath(os.path.join(run_dir, f"{sample}.mzML"))
        oms.MzMLFile().store(os.path.join(run_dir, f"{sample}.mzML"), merged_exp)
    runtimes["over-file merging"] = [time.time() - step_time]
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
