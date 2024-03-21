import sys

from MStoML.skripts.FIA.FIA import load_fia_df
sys.path.append( '../FIA' )

from FIA import *

test_experiments = ""   # TODO

fia_df = load_fia_df(test_experiments, ".mzML")
test_df1 = fia_df["experiment"][0].get_df(long=True)

def ml4com_tests():
    assert np.sum(test_df1["inty"]) == bin_df_stepwise(test_df1, binning_var="mz", binned_var="inty", statistic="sum",
                                                       start=np.min(test_df1["inty"]), stop=np.max(test_df1["inty"]), step=0.001)["inty"]

def main():
    print("Testing ML4com...")
    ml4com_tests()

