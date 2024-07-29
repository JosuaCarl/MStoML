import numpy as np
import pandas as pd

def total_ion_count_normalization(df, axis=0):
    """
    Perform TIC normalization on dataframe.

    :param df: Dataframe
    :type df: pandas.DataFrame
    :param axis: Axis to perform normalization on, defaults to 0
    :type axis: int, optional
    :return: Normalized Dataframe
    :rtype: pandas.DataFrame
    """    
    return df.div( df.sum(axis=axis), axis=1 if axis == 0 else 0)

def standard_normalization(df, axis=1):
    """
    Perform standard normalization (centered around 0, scaled to variance) on dataframe.

    :param df: Dataframe
    :type df: pandas.DataFrame
    :param axis: Axis to perform normalization on, defaults to 0
    :type axis: int, optional
    :return: Normalized Dataframe
    :rtype: pandas.DataFrame
    """    
    return df.apply(lambda line: [(x - np.mean(line)) / np.var(line) for x in line], result_type="expand", axis=axis)
