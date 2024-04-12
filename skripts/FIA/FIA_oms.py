# Imports
import os
import time
import sys
import argparse

# import methods from FIA python script
sys.path.append("..")
from FIA.FIA import *

# Argument parser
parser = argparse.ArgumentParser(prog='FIA_oms',
                                description='Flow-injection analysis with OpenMS python bindings.')
parser.add_argument('-d', '--data_dir', nargs=1, required=True)      # option that takes a value
parser.add_argument('-r', '--run_dir', nargs=1, required=True)
parser.add_argument('-s', '--steps', nargs="*", required=True)
args = parser.parse_args()

def main():
    data_dir, run_dir = (args.data_dir, args.run_dir)
    data_dir = os.path.normpath(os.path.join(os.getcwd(), data_dir))
    run_dir = os.path.normpath(os.path.join(os.getcwd(), run_dir))
    steps = args.steps if args.steps else ["trim", "centroid", "merge", "pos_neg_merge"]

    runner = Runner(steps, data_dir, run_dir, runtimes={}, start_time=time.time())
    methods = {"trim": runner.trim(),
               "centroid": runner.centroid(),
               "merge":runner.merge(),
               "pos_neg_merge": runner.pos_neg_merge(),
               "save_runtimes": runner.save_runtimes()}
    steps = [methods[step] for step in steps]
    print("Finished")


class Runner:
    def __init__(self, steps, data_dir, run_dir, runtimes, start_time) -> None:
        self.steps = steps
        self.run_dir = run_dir
        self.last_dir = data_dir
        self.runtimes = runtimes
        self.start_time = start_time
        self.step_time = start_time
        self.merge_dict = {}


    def trim(self):
        print("Trim values:")
        self.last_dir = trim_threshold_batch(self.last_dir, self.run_dir, file_ending=".mzML", threshold=1e3, deepcopy=False)
        self.runtimes["trimming"] = [time.time() - self.step_time]
        self.step_time = time.time()
        return True
        

    def centroid(self):
        print("Centroiding:")
        self.last_dir = centroid_batch(self.last_dir, self.run_dir, file_ending=".mzML", instrument="TOF",
                                        signal_to_noise=2.0, spacing_difference=1.5,
                                        peak_width=0.0, sn_bin_count=100, nr_iterations=5, sn_win_len=20.0,
                                        check_width_internally="false", ms1_only="true", clear_meta_data="false",
                                        deepcopy=False )
        self.runtimes["centroiding"] = [time.time() - self.step_time]
        self.step_time = time.time()
        return True


    def merge(self):
        print("Merging:")
        self.last_dir = merge_batch(self.last_dir, self.run_dir, file_ending=".mzML", method="block_method",
                                mz_binning_width=10.0, mz_binning_width_unit="ppm",
                                ms_levels=[1], sort_blocks="RT_ascending",
                                rt_block_size=None, rt_max_length=0.0,
                                )
        self.runtimes["merging"] = [time.time() - self.step_time]
        self.step_time = time.time()
        return True


    def pos_neg_merge(self):
        print("Sample merging:")
        self.merge_dict = make_merge_dict(self.last_dir, file_ending=".mzML")
        for sample, files in self.merge_dict.items():
            merged_exp = merge_experiments(files, self.run_dir, file_ending=".mzML", method="block_method",
                                            mz_binning_width=10.0, mz_binning_width_unit="ppm",
                                            ms_levels=[1], sort_blocks="RT_ascending",
                                            rt_block_size=None, rt_max_length=0.0 )
            merged_exp.setLoadedFilePath(os.path.join(self.run_dir, f"{sample}.mzML"))
            oms.MzMLFile().store(os.path.join(self.run_dir, f"{sample}.mzML"), merged_exp)
        self.runtimes["over-file merging"] = [time.time() - self.step_time]
        self.step_time = time.time()
        return True


    def save_runtimes(self):
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
