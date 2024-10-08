#!/usr/bin/env python3
"""
Flow injection analysis with OpenMS.
"""

# Imports
import os
import time
import sys
import argparse

# import methods from FIA python script
sys.path.append("..")
from FIA.FIA import *

def main():
    """
    Start shell script
    """    
    # Argument parser
    parser = argparse.ArgumentParser(prog='FIA_oms',
                                     description='Flow-injection analysis with OpenMS python bindings.')
    parser.add_argument('-d', '--data_dir', required=True)      # option that takes a value
    parser.add_argument('-e', '--file_ending', required=False)
    parser.add_argument('-r', '--run_dir', required=True)
    parser.add_argument('-s', '--steps', nargs="*", required=True)
    args = parser.parse_args()

    data_dir, run_dir = (args.data_dir, args.run_dir)
    data_dir = os.path.normpath(os.path.join(os.getcwd(), data_dir))
    run_dir = os.path.normpath(os.path.join(os.getcwd(), run_dir))
    if not os.path.isdir(run_dir):
        os.mkdir(run_dir)
    file_ending = args.file_ending if args.file_ending else ".mzML"
    steps = args.steps if args.steps else ["trim", "centroid", "merge", "pos_neg_merge"]

    runner = Runner(data_dir, run_dir, file_ending, runtimes={}, start_time=time.time())
    methods = {"trim": runner.trim,
               "centroid": runner.centroid,
               "merge":runner.merge,
               "pos_neg_merge": runner.pos_neg_merge,
               "save_runtimes": runner.save_runtimes}
    steps = [methods[step]() for step in steps]
    print("Finished")


class Runner:
    """
    Class for running FIA analysis via pyopenms.
    """    
    def __init__(self, data_dir, run_dir, file_ending:str, runtimes:dict, start_time) -> None:
        """
        Initiate runner.

        :param data_dir: Data directory
        :type data_dir: path-like
        :param run_dir: Run directory
        :type run_dir: path-like
        :param file_ending: File ending
        :type file_ending: str
        :param runtimes: Runtimes dictionary
        :type runtimes: dict
        :param start_time: Starting time
        :type start_time: time
        """        
        self.run_dir = run_dir
        self.last_dir = data_dir
        self.file_ending = file_ending
        self.runtimes = runtimes
        self.start_time = start_time
        self.step_time = start_time
        self.merge_dict = {}


    def trim(self) -> bool:
        """
        Trim batch according to threshold

        :return: Success of operation
        :rtype: bool
        """        
        print("Trim values:")
        self.last_dir = trim_threshold_batch(self.last_dir, self.run_dir, file_ending=self.file_ending, threshold=1e3, deepcopy=False)
        self.file_ending = ".mzML"
        self.runtimes["trimming"] = [time.time() - self.step_time]
        self.step_time = time.time()
        return True
        

    def centroid(self):
        """
        Centroid batch according to parameters in script.

        :return: Success of operation
        :rtype: bool
        """
        print("Centroiding:")
        self.last_dir = centroid_batch(self.last_dir, self.run_dir, file_ending=self.file_ending, instrument="TOF",
                                        signal_to_noise=2.0, spacing_difference=1.5,
                                        peak_width=0.0, sn_bin_count=100, nr_iterations=5, sn_win_len=20.0,
                                        check_width_internally="false", ms1_only="true", clear_meta_data="false",
                                        deepcopy=False )
        self.file_ending = ".mzML"
        self.runtimes["centroiding"] = [time.time() - self.step_time]
        self.step_time = time.time()
        return True


    def merge(self):
        """
        Merge batch according to parameters in script.

        :return: Success of operation
        :rtype: bool
        """
        print("Merging:")
        self.last_dir = merge_batch(self.last_dir, self.run_dir, file_ending=self.file_ending, method="block_method",
                                mz_binning_width=10.0, mz_binning_width_unit="ppm",
                                ms_levels=[1], sort_blocks="RT_ascending",
                                rt_block_size=None, rt_max_length=0.0,
                                )
        self.file_ending = ".mzML"
        self.runtimes["merging"] = [time.time() - self.step_time]
        self.step_time = time.time()
        return True


    def pos_neg_merge(self):
        """
        Merge batch along positive and negative ionization modes according to parameters in script.

        :return: Success of operation
        :rtype: bool
        """
        print("Sample merging:")
        self.merge_dict = make_merge_dict(self.last_dir, file_ending=self.file_ending)
        for sample, files in self.merge_dict.items():
            merged_exp = merge_experiments(files, self.run_dir, file_ending=self.file_ending, method="block_method",
                                            mz_binning_width=10.0, mz_binning_width_unit="ppm",
                                            ms_levels=[1], sort_blocks="RT_ascending",
                                            rt_block_size=None, rt_max_length=0.0 )
            merged_exp.setLoadedFilePath(os.path.join(self.run_dir, f"{sample}.mzML"))
            oms.MzMLFile().store(os.path.join(self.run_dir, f"{sample}.mzML"), merged_exp)
        self.file_ending = ".mzML"
        self.runtimes["over-file merging"] = [time.time() - self.step_time]
        self.step_time = time.time()
        return True


    def save_runtimes(self):
        """
        Save the runtimes.

        :return: Success of operation
        :rtype: bool
        """        
        runtime = time.time() - self.start_time
        self.runtimes["total"] = [runtime]
        runtimes = pd.DataFrame(self.runtimes)
        runtimes.loc[1] = np.array(runtimes.loc[runtimes.index[0]].values) / len(self.merge_dict.keys())
        runtimes.loc[2] = np.array(runtimes.loc[runtimes.index[0]].values) / len(sum(self.merge_dict.values(), []))
        runtimes.index = pd.Index( ["total", "per sample", "per file"] )
        runtimes.to_csv(os.path.join(self.run_dir, "runtimes.tsv"), sep="\t")
        print(f"Runtime: {int(runtime)} s")
        return True



if __name__ == "__main__":
    main()
