import numpy as np
import pandas as pd

def total_ion_count_normalization(df, axis=0):
    return df.div( df.sum(axis=axis), axis=1 if axis == 0 else 0)

def standard_normalization(df, axis=1):
    return df.apply(lambda line: [(x - np.mean(line)) / np.var(line) for x in line], result_type="expand", axis=axis)
