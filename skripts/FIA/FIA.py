#!/usr/bin/env python3
"""
Methods for Flow-injection-analysis.
"""

# Imports
import os
import gc
import time
from typing import List, Tuple, Sequence, Union, Optional
import shutil
import requests
from copy import deepcopy
from tqdm import tqdm
import psutil

import numpy as np
import pandas as pd
import polars as pl

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

import pyopenms as oms

import scipy as sci
from scipy.signal import find_peaks
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import fcluster, linkage

from sklearn.impute import KNNImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline


## Data
def batch_download(base_url:str, file_urls:list, save_directory:str) -> None:
    """
    Download files from a list into a directory.

    :param base_url: Base URL
    :type base_url: str
    :param file_urls: Individual file URLS which are appended to the base
    :type file_urls: list
    :param save_directory: Directory to save files to.
    :type save_directory: str
    """    
    for file_url in file_urls:
        request = requests.get(base_url + file_url, allow_redirects=True)
        open(os.path.join(save_directory, os.path.basename(file_url)), "wb").write(request.content)


# Helpers
def bits_to_bytes(bits:Union[float, int], factor:Union[float, int]) -> float:
    """
    Coverts a number of bits to a number of bytes for readability.

    :param bits: Number of bits to be converted
    :type bits: Union[float, int]
    :param factor: / 10**factor (e.g. use 9 for GB)
    :type factor: Union[float, int]
    :return: Number of bytes
    :rtype: float
    """
    return round((bits * 0.125) / 10**factor, 5)



## Directories
def build_directory(dir_path:str) -> None:
    """
    Build a new directory in the given path.

    :param dir_path: Directory path
    :type dir_path: str
    """    
    if not os.path.isdir(os.path.join(os.getcwd(), dir_path)):
        os.mkdir(os.path.join(os.getcwd(), dir_path))

def clean_dir(dir_path:str, subfolder:Optional[str]=None) -> str:
    """
    Delete a directory or its subfolder and reconstruct the directory.

    :param dir_path: Directory path
    :type dir_path: str
    :param subfolder: Subfolder path, defaults to None
    :type subfolder: Optional[str], optional
    :return: Directory path
    :rtype: str
    """    
    if subfolder:
        dir_path = os.path.join(dir_path, subfolder)
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.mkdir(dir_path)
    return dir_path



## Files
### Checking
def check_ending_experiment(file:str) -> bool:
    """
    Check whether the file has a mzML or mzXML ending.

    :param file: Path to file
    :type file: str
    :return: Ending is mzML or mzXML
    :rtype: bool
    """    
    return file.endswith(".mzML") or file.endswith(".MzML") or file.endswith(".mzXML") or file.endswith(".MzXML")



### Loading
def read_experiment(experiment_path: str, separator:str="\t") -> oms.MSExperiment:
    """
    Read in MzXML or MzML File as a pyopenms experiment. If the file is in tabular format,
    assumes that is is in long form with two columns ["mz", "inty"]

    :param experiment_path: Path to experiment
    :type experiment_path: str
    :param separator: Separator of data, defaults to "\t"
    :type separator: str, optional
    :raises ValueError: The experiment must end with a valid ending.
    :return: Experiment
    :rtype: pyopenms.MSExperiment
    """    
    experiment = oms.MSExperiment()
    if experiment_path.endswith(".mzML") or experiment_path.endswith(".MzML"):
        file = oms.MzMLFile()
        file.load(experiment_path, experiment)
    elif experiment_path.endswith(".mzXML") or experiment_path.endswith(".MzXML"):
        file = oms.MzXMLFile()
        file.load(experiment_path, experiment)
    elif experiment_path.endswith(".tsv") or experiment_path.endswith(".csv") or experiment_path.endswith(".txt"):
        exp_df = pd.read_csv(experiment_path, sep=separator)
        spectrum = oms.MSSpectrum()
        spectrum.set_peaks( (exp_df["mz"], exp_df["inty"]) ) # type: ignore
        experiment.addSpectrum(spectrum)
    else: 
        raise ValueError(f'Invalid ending of {experiment_path}. Must be in [".MzXML", ".mzXML", ".MzML", ".mzML", ".tsv", ".csv", ".txt"]')
    return experiment


def load_experiment(experiment:Union[oms.MSExperiment, str], separator:str="\t") -> oms.MSExperiment:
    """
    If no experiment is given, loads and returns it from either .mzML or .mzXML file.
    Collects garbage with gc.collect() to ensure space in the RAM.

    :param experiment: Experiment, or Path to experiment
    :type experiment: Union[oms.MSExperiment, str]
    :param separator: Separator of data, defaults to "\t"
    :type separator: str, optional
    :return: Experiment
    :rtype: pyopenms.MSExperiment
    """
    gc.collect()
    if isinstance(experiment, oms.MSExperiment):
        return experiment
    else:
        return read_experiment(experiment, separator=separator)
    
def load_experiments(experiments:Union[Sequence[Union[oms.MSExperiment,str]], str], file_ending:Optional[str]=None,
                     separator:str="\t", data_load:bool=True) -> Sequence[Union[oms.MSExperiment,str]]:
    """
    Load a batch of experiments.

    :param experiments: Experiments, either described by a list of paths or one path as base directory,
    or an existing experiment.
    :type experiments: Union[Sequence[Union[oms.MSExperiment,str]], str]
    :param file_ending: Ending of experiment file, defaults to None
    :type file_ending: Optional[str], optional
    :param separator: Separator of data, defaults to "\t"
    :type separator: str, optional
    :param data_load: Load the data or just combine the base string to a list of full filepaths, defaults to True
    :type data_load: bool, optional
    :return: Experiments
    :rtype: Sequence[Union[oms.MSExperiment,str]]
    """    
    """
    
    """
    if isinstance(experiments, str):
        if file_ending:
            experiments = [os.path.join(experiments, file) for file in os.listdir(experiments) if file.endswith(file_ending)]
        else:
            experiments = [os.path.join(experiments, file) for file in os.listdir(experiments) if check_ending_experiment(file)]
    if data_load:
        experiments = [load_experiment(experiment, separator=separator) for experiment in tqdm(experiments)]
    return experiments


def load_name(experiment:Union[oms.MSExperiment, str], alt_name:Optional[str]=None, file_ending:Optional[str]=None) -> str:
    """
    Load the name of an experiment.

    :param experiment: Experiment
    :type experiment: Union[oms.MSExperiment, str]
    :param alt_name: Alternative Name if none is found, defaults to None
    :type alt_name: Optional[str], optional
    :param file_ending: Ending of experiment file, defaults to None
    :type file_ending: Optional[str], optional
    :raises ValueError: Raises error if no file name is found and no alt_name is given.
    :return: Name of experiment or alternative name
    :rtype: str
    """    
    if isinstance(experiment, str):
        return "".join(experiment.split(".")[:-1])
    else:
        if experiment.getLoadedFilePath():
            return "".join(os.path.basename(experiment.getLoadedFilePath()).split(".")[:-1])
        elif alt_name:
            return alt_name
        else:
            raise ValueError(f"No file path found in experiment. Please provide alt_name.")

def load_names_batch(experiments:Union[Sequence[Union[oms.MSExperiment,str]], str], file_ending:str=".mzML") -> List[str]:
    """
    If no experiment is given, loads and returns it from either .mzML or .mzXML file.

    :param experiments: Experiments
    :type experiments: Union[Sequence[Union[oms.MSExperiment,str]], str]
    :param file_ending: Ending of experiment file, defaults to ".mzML"
    :type file_ending: str, optional
    :return: List of experiment names
    :rtype: List[str]
    """    
    if isinstance(experiments, str):
        if file_ending:
            return [load_name(file) for file in tqdm(os.listdir(experiments)) if file.endswith(file_ending)]
        else:
            return [load_name(file) for file in tqdm(os.listdir(experiments)) if check_ending_experiment(file)]
    else:
        if isinstance(experiments[0], str):
            return [load_name(experiment) for experiment in tqdm(experiments)] 
        else:
            return [load_name(experiment, str(i)) for i, experiment in enumerate(tqdm(experiments))]
    

def load_fia_df(data_dir:str, file_ending:str, separator:str="\t", data_load:bool=True, backend=pd) -> Union[pd.DataFrame, pl.DataFrame]:
    """
    Load a Flow injection analysis dataframe, defining important properties.

    :param data_dir: Data directory
    :type data_dir: str
    :param file_ending: Ending of file
    :type file_ending: str
    :param separator: Separator for file, defaults to "\t"
    :type separator: str, optional
    :param data_load: Load data or only return list of experiments, defaults to True
    :type data_load: bool, optional
    :param backend: Use pandas or polars as backend, defaults to pd
    :type backend: _type_, optional
    :return: _description_
    :rtype: Union[pandas.DataFrame, polars.DataFrame]
    """    
    print("Loading names:")
    names = load_names_batch(data_dir, file_ending)
    samples = [name.split("_")[0] for name in names]
    polarities = [{"pos": 1, "neg": -1}.get(name.split("_")[-1]) for name in names]
    print("Loading experiments:")
    experiments = load_experiments(data_dir, file_ending, separator=separator, data_load=data_load)
    fia_df = backend.DataFrame([samples, polarities, experiments])
    fia_df = fia_df.transpose()
    fia_df.columns = ["sample", "polarity", "experiment"]
    return fia_df


def read_mnx(filepath: str) -> pd.DataFrame:
    """
    Read in chem_prop.tsv file from MetaNetX

    :param filepath: Path to file
    :type filepath: str
    :return: DataFrame
    :rtype: pd.DataFrame
    """
    return pd.read_csv(filepath, sep="\t",
                       header=351, engine="pyarrow"
                       )[["#ID", "name", "formula", "charge", "mass"]].loc[1:].reset_index(drop=True).dropna()


def read_feature_map_XML(path_to_featureXML:str) -> oms.FeatureMap:
    """
    Reads in feature Map from .featureXML file.

    :param path_to_featureXML: Path to featureXML file.
    :type path_to_featureXML: str
    :return: Feature map
    :rtype: pyopenms.FeatureMap
    """
    fm = oms.FeatureMap()
    fh = oms.FeatureXMLFile()
    fh.load(path_to_featureXML, fm)
    return fm

def read_feature_maps_XML(path_to_featureXMLs:str) -> list:
    """
    Reads in feature Maps from file

    :param path_to_featureXMLs: Path to featureXML file directory.
    :type path_to_featureXMLs: str
    :return: List of feature maps
    :rtype: list
    """    
    feature_maps = []
    print("Reading in feature maps:")
    for file in tqdm(os.listdir(path_to_featureXMLs)):
        fm = read_feature_map_XML(os.path.join(path_to_featureXMLs, file))
        feature_maps.append(fm)
    return feature_maps


def define_metabolite_table(path_to_library_file:str, mass_range:list) -> list:
    """
    Read tsv file and create list of FeatureFinderMetaboIdentCompound.

    :param path_to_library_file: Path to library file
    :type path_to_library_file: str
    :param mass_range: Range of m/z values
    :type mass_range: list
    :return: Metabolite table
    :rtype: list
    """    
    metabo_table = []
    df = pd.read_csv(path_to_library_file, quotechar='"', sep="\t")
    print(f"Read in {len(df.index)} metabolites.")
    df = df.loc[[False in np.isin(list(map(int, row["Charge"][1:-1].split(","))), np.zeros(len(row["Charge"]))) for i, row in df.iterrows()]]
    print(f"{len(df.index)} remaining after excluding zero charged metabolites.")
    for i, row in df.iterrows(): 
        metabo_table.append(
            oms.FeatureFinderMetaboIdentCompound(
                row["CompoundName"], row["SumFormula"], row["Mass"],
                list(map(int, row["Charge"][1:-1].split(","))), 
                list(map(float, row["RetentionTime"][1:-1].split(","))),
                list(map(float, row["RetentionTimeRange"][1:-1].split(","))),
                list(map(float, row["IsotopeDistribution"][1:-1].split(",")))
            )
        )
    print("Finished metabolite table.")
  
    return metabo_table



### Copying
def copy_experiment(experiment: oms.MSExperiment) -> oms.MSExperiment:
    """
    Makes a complete (recursive) copy of an experiment.

    :param experiment: Experiment
    :type experiment: pyopenms.MSExperiment
    :return: Copy of experiment.
    :rtype: pyopenms.MSExperiment
    """
    return deepcopy(experiment)



### Formatting
def mnx_to_oms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Turns a dataframe from MetaNetX into the required format by pyopenms for feature detection.

    :param df: Parameters Dataframe
    :type df: pandas.DataFrame
    :return: DataFrame with essential information.
    :rtype: pandas.DataFrame
    """
    return pd.DataFrame(list(zip(df["name"].values,
                                 df["formula"].values,
                                 df["mass"].values,
                                 df["charge"].values,
                                 np.ones(len(df.index)),
                                 np.zeros(len(df.index)),
                                 np.zeros(len(df.index)))),
                        columns=["CompoundName", "SumFormula", "Mass", "Charge", "RetentionTime", "RetentionTimeRange",
                                 "IsotopeDistribution"])


def join_df_by(df: pd.DataFrame, joiner: str, combiner: str) -> pd.DataFrame:
    """
    Combines datframe with same <joiner>, while combining the name of <combiner> as the new index.

    :param df: Input DataFrame
    :type df: pd.DataFrame
    :param joiner: Indicates the column that is the criterium for joining the rows
    :type joiner: str
    :param combiner: Indicates the column that should be combined as an identifier
    :type combiner: str
    :return: Combined DataFrame
    :rtype: pd.DataFrame
    """ 
    comb = pd.DataFrame(columns=df.columns)
    for j in tqdm(df[joiner].unique()):
        query = df.loc[df[joiner] == j]
        line = query.iloc[0].copy()
        line[combiner] = query[combiner].to_list()
        comb.loc[len(comb.index)] = line
    comb = comb.set_index(combiner)
    return comb



### Annotation
def annotate_consensus_map_df(consensus_map_df:pd.DataFrame, mass_search_df:pd.DataFrame, result_path:str=".",
                              mz_tolerance:float=1e-05) -> pd.DataFrame:
    """
    Annotate consensus map DataFrame.

    :param consensus_map_df: Input Consensus map DataFrame
    :type consensus_map_df: pd.DataFrame
    :param mass_search_df: Mass search DataFrame
    :type mass_search_df: pd.DataFrame
    :param result_path: Path to output results, defaults to "."
    :type result_path: str, optional
    :param mz_tolerance: Tolerance of m/z deviation, defaults to 1e-05
    :type mz_tolerance: float, optional
    :return: Identified Metabolites DataFrame
    :rtype: pd.DataFrame
    """    
    id_df = consensus_map_df[["mz", "centroided_intensity"]].copy()

    id_df["identifier"] = pd.Series([""]*len(id_df.index))

    for mz, identifier in zip(mass_search_df["exp_mass_to_charge"],
                              mass_search_df["identifier"],):
        indices = id_df.index[np.isclose(id_df["mz"], float(mz), mz_tolerance)].tolist()
        for index in indices:
            id_df.loc[index, "identifier"] += str(identifier) + ";"             # type: ignore
    id_df["identifier"] = [item[:-1] if ";" in item else "" for item in id_df["identifier"]]

    id_df.to_csv(result_path, sep="\t", index=False)
    return id_df



# Storing
def store_experiment(experiment_path:str, experiment: oms.MSExperiment) -> None:
    """
    Stores the experiment.

    :param experiment_path: Path to be stored at (must end with .mzMl or .mzXML)
    :type experiment_path: str
    :param experiment: Experiment
    :type experiment: pyopenms.MSExperiment
    """
    if experiment_path.endswith(".mzXML"):
        oms.MzXMLFile().store(experiment_path, experiment)
    elif experiment_path.endswith(".mzML"):
        oms.MzMLFile().store(experiment_path, experiment)
    else:
        oms.MzMLFile().store(experiment_path, experiment)


def store_feature_maps(feature_maps: list, out_dir:str, names: Union[list[str], str]=[], file_ending:str=".mzML") -> None:
    """
    Stores the feature maps as featureXML files.

    :param feature_maps: Feature Maps
    :type feature_maps: list
    :param out_dir: Output directory
    :type out_dir: str
    :param names: Names of feature maps, defaults to []
    :type names: Union[list[str], str], optional
    :param file_ending: Ending of file, defaults to ".mzML"
    :type file_ending: str, optional
    """    
    clean_dir(out_dir)
    print("Storing feature maps:")
    if isinstance(names, str):
        names = [os.path.basename(file)[:-len(file_ending)] for file in os.listdir(names) if file.endswith(file_ending)]
    for i, feature_map in enumerate(tqdm(feature_maps)):
        if names:
            name = names[i]
        else:
            if feature_map.getMetaValue("spectra_data"):
                name = os.path.basename(feature_map.getMetaValue("spectra_data")[0].decode())[:-len(file_ending)]
            else:
                name = f"features_{i}"
        oms.FeatureXMLFile().store(os.path.join(out_dir, name + ".featureXML"), feature_map)



## Compound tables transformation
def merge_compounds(path_to_tsv:str) -> pd.DataFrame:
    """
    Joins entries with equal Mass and SumFormula. Links CompoundName with: ;. Links rest with: ,.

    :param path_to_tsv: Path to TSV
    :type path_to_tsv: str
    :return: Merged compounds
    :rtype: pd.DataFrame
    """
    df = pd.read_csv(path_to_tsv, sep="\t")
	
    aggregation_functions = {'CompoundName': lambda x: ";".join(x)}
    groupies = [ df['Mass'], df["SumFormula"], df["Charge"], df["RetentionTime"], df["RetentionTimeRange"], df["IsotopeDistribution"] ]
    df_new = df.groupby( groupies , as_index=False).aggregate(aggregation_functions)


    aggregation_functions = {'Charge': lambda x: x.to_list(),
                                'RetentionTime': lambda x: x.to_list(),
                                'RetentionTimeRange': lambda x: x.to_list(),
                                'IsotopeDistribution': lambda x: x.to_list()}
    groupies = [ df_new['Mass'], df_new["SumFormula"], df_new["CompoundName"]]
    df_new = df_new.groupby( groupies , as_index=False).aggregate(aggregation_functions)

    # Exclude inorganics
    organic_elements = ["H", "C", "N", "P", "I", "Cu", "Mg", "Na", "K", "Zn", "S", "Ca", "Co", "Fe", "O"]
    elements = ['Ac', 'Ag', 'Al', 'Am', 'Ar', 'As', 'At', 'Au', 'B', 'Ba', 'Be', 'Bh', 'Bi', 'Bk', 'Br', 'C', 'Ca', 'Cd', 'Ce', 'Cf', 'Cl', 'Cm', 'Cn', 'Co', 'Cr', 'Cs', 'Cu', 'Db', 'Ds', 'Dy', 'Er', 'Es', 'Eu', 'F', 'Fe', 'Fl', 'Fm', 'Fr', 'Ga', 'Gd', 'Ge', 'H', 'He', 'Hf', 'Hg', 'Ho', 'Hs', 'I', 'In', 'Ir', 'K', 'Kr', 'La', 'Li', 'Lr', 'Lu', 'Lv', 'Mc', 'Md', 'Mg', 'Mn', 'Mo', 'Mt', 'N', 'Na', 'Nb', 'Nd', 'Ne', 'Nh', 'Ni', 'No', 'Np', 'O', 'Og', 'Os', 'P', 'Pa', 'Pb', 'Pd', 'Pm', 'Po', 'Pr', 'Pt', 'Pu', 'Ra', 'Rb', 'Re', 'Rf', 'Rg', 'Rh', 'Rn', 'Ru', 'S', 'Sb', 'Sc', 'Se', 'Sg', 'Si', 'Sm', 'Sn', 'Sr', 'Ta', 'Tb', 'Tc', 'Te', 'Th', 'Ti', 'Tl', 'Tm', 'Ts', 'U', 'V', 'W', 'Xe', 'Y', 'Yb', 'Zn', 'Zr']
    exclude_elements = [e for e in elements if e not in organic_elements]
    for element in exclude_elements:
        df_new = df_new.loc[~df_new["SumFormula"].str.contains(element)]

    return df_new[["CompoundName", "SumFormula", "Mass", "Charge", "RetentionTime", "RetentionTimeRange", "IsotopeDistribution"]].reset_index(drop=True)



## MS data transformation
### Binning
def bin_df_stepwise(df:Union[pd.DataFrame, pl.DataFrame], binning_var="mz", binned_var="inty", statistic="sum",
                    start:float=0.0, stop:float=2000.0, step:float=0.001, backend=pd) -> Union[pd.DataFrame, pl.DataFrame]:
    """
    Stepwise binning of a dataframe into discrete boxes.

    :param df: Input dataframe
    :type df: Union[pd.DataFrame, pl.DataFrame]
    :param binning_var: Binning variable, i.e. distance defining value (column in dataframe), defaults to "mz"
    :type binning_var: str, optional
    :param binned_var: Binned value, i.e. combined value (column in dataframe), defaults to "inty"
    :type binned_var: str, optional
    :param statistic: Operation to perform on binned values in a window. May take common parameters,
    defined by sci.stats.binned_statistic. defaults to "sum"
    :type statistic: str, optional
    :param start: Starting binning variable point, defaults to 0.0
    :type start: float, optional
    :param stop: Stopping binning variable point, defaults to 2000.0
    :type stop: float, optional
    :param step: Step distance along binning variable, defaults to 0.001
    :type step: float, optional
    :param backend: Backend (pandas or polars), defaults to pd
    :type backend: _type_, optional
    :return: Binned dataframe
    :rtype: Union[pd.DataFrame, pl.DataFrame]
    """    
    bins = np.append(np.arange(start, stop, step), stop)
    statistic, bin_edges, bin_nrs = sci.stats.binned_statistic(df[binning_var], df[binned_var],
                                                               statistic=statistic, bins=bins, range=(start, stop))
    bin_means = np.mean([bins[1:], bins[:-1]], axis=0)
                         
    binned_df = backend.DataFrame({"mz": bin_means, "inty": statistic})
    binned_df.set_index("mz", inplace=True) 
    return binned_df

def bin_df_stepwise_batch(experiments:Union[pd.DataFrame, pl.DataFrame],
                          sample_var:str="sample", experiment_var:str="experiment",
                          binning_var="mz", binned_var="inty", statistic="sum",
                          start:float=0.0, stop:float=2000.0, step:float=0.001,
                          backend=pd) -> Union[pd.DataFrame, pl.DataFrame]:
    """
    Stepwise binning of a dataframe into discrete boxes of multiple dataframes.

    :param experiments: Input experiments
    :type experiments: Union[pd.DataFrame, pl.DataFrame]
    :param binning_var: Binning variable, i.e. distance defining value (column in dataframe), defaults to "mz"
    :type binning_var: str, optional
    :param binned_var: Binned value, i.e. combined value (column in dataframe), defaults to "inty"
    :type binned_var: str, optional
    :param statistic: Operation to perform on binned values in a window. May take common parameters,
    defined by sci.stats.binned_statistic. defaults to "sum"
    :type statistic: str, optional
    :param start: Starting binning variable point, defaults to 0.0
    :type start: float, optional
    :param stop: Stopping binning variable point, defaults to 2000.0
    :type stop: float, optional
    :param step: Step distance along binning variable, defaults to 0.001
    :type step: float, optional
    :param backend: Backend (pandas or polars), defaults to pd
    :type backend: _type_, optional
    :return: Binned dataframe
    :rtype: Union[pd.DataFrame, pl.DataFrame]
    """    
    binned_dfs = backend.DataFrame()
    for i, row in tqdm(experiments.iterrows(), total=len(experiments)):
        experiment = row[experiment_var]
        if not isinstance(experiment, backend.DataFrame):
            experiment = experiment.get_df(long=True)
        binned_df = bin_df_stepwise(experiment, binning_var=binning_var, binned_var=binned_var,
                                    statistic=statistic, start=start, stop=stop, step=step,
                                    backend=backend)
        if binned_dfs.empty:
            binned_dfs = binned_df
        else:
            binned_dfs = binned_dfs.join(binned_df)
        binned_dfs.rename(columns={binned_var: row[sample_var]}, inplace=True)
    return binned_dfs



# Limiting
def limit_spectrum(spectrum: oms.MSSpectrum, mz_lower_limit: Union[int, float], mz_upper_limit: Union[int, float],
                   sample_size: int, statistic:str="sum") -> oms.MSSpectrum:
    """
    Limits the range of the Spectrum to <mz_lower_limit> and <mz_upper_limit>. 
    Uniformly samples <sample_size> number of peaks from the spectrum (without replacement).

    :param spectrum: Input spectrum
    :type spectrum: pyopenms.MSSpectrum
    :param mz_lower_limit: Lower m/z value limit
    :type mz_lower_limit: Union[int, float]
    :param mz_upper_limit: Upper m/z value limit
    :type mz_upper_limit: Union[int, float]
    :param sample_size: Number of sampled peaks from the spectrum
    :type sample_size: int
    :param statistic: Operation to perform on binned values in a window. May take common parameters,
    defined by sci.stats.binned_statistic., defaults to "sum"
    :type statistic: str, optional
    :return: Limited and sampled spectrum
    :rtype: pyopenms.MSSpectrum
    """    
    new_spectrum = oms.MSSpectrum()
    mzs, intensities = spectrum.get_peaks() # type: ignore

    if statistic:
        statistic, bin_edges, bin_nrs = sci.stats.binned_statistic(mzs, intensities,
                                                                   statistic=statistic, 
                                                                   bins=sample_size,
                                                                   range=(mz_lower_limit, mz_upper_limit))
        statistic = np.nan_to_num(statistic)
        bin_means = np.mean([bin_edges[1:], bin_edges[:-1]], axis=0)
        new_spectrum.set_peaks( (bin_means, statistic) ) # type: ignore
    else:
        lim = [np.searchsorted(mzs, mz_lower_limit, side='right'), np.searchsorted(mzs, mz_upper_limit, side='left')]
        mzs = mzs[lim[0]:lim[1]]
        intensities = intensities[lim[0]:lim[1]]

        idxs = range(len(mzs))
        if len(mzs) > sample_size:
            idxs = np.random.choice(idxs, size=sample_size, replace=False)
            new_spectrum.set_peaks( (mzs[idxs], intensities[idxs]) ) # type: ignore
        else:
            new_spectrum = spectrum
    return new_spectrum


def limit_experiment(experiment: Union[oms.MSExperiment, str], mz_lower_limit: Union[int, float]=0, mz_upper_limit: Union[int, float]=10000,
                     sample_size:int=100000, statistic:str="sum", deepcopy: bool = False) -> oms.MSExperiment:
    """
    Limits the range of all spectra in an experiment to <mz_lower_limit> and <mz_upper_limit>. 
    Uniformly samples <sample_size> number of peaks from the spectrum (without replacement).

    :param experiment: Input experiment
    :type experiment: Union[pyopenms.MSExperiment, str]
    :param mz_lower_limit: Lower m/z value limit, defaults to 0
    :type mz_lower_limit: Union[int, float], optional
    :param mz_upper_limit: Upper m/z value limit, defaults to 10000
    :type mz_upper_limit: Union[int, float], optional
    :param sample_size: Number of sampled peaks from the spectrum, defaults to 100000
    :type sample_size: int, optional
    :param statistic: Operation to perform on binned values in a window. May take common parameters,
    defined by sci.stats.binned_statistic., defaults to "sum"
    :type statistic: str, optional
    :param deepcopy: Perform a deepcopy to relieably unlink the experiment from the input, defaults to False
    :type deepcopy: bool, optional
    :return: Limited and sampled experiment
    :rtype: oms.MSExperiment
    """
    experiment = load_experiment(experiment)

    if deepcopy:
        lim_exp = copy_experiment(experiment)
    else:
        lim_exp = experiment
    lim_exp.setSpectra(
        [limit_spectrum(spectrum, mz_lower_limit, mz_upper_limit, sample_size, statistic) for spectrum in experiment.getSpectra()])
    return lim_exp



def trim_threshold(experiment:oms.MSExperiment, threshold:float=0.05) -> oms.Experiment:
    """
    Removes point below an absolute intensity theshold.

    :param experiment: Input Experiment
    :type experiment: pyopenms.MSExperiment
    :param threshold: Threshold for values to be excluded, defaults to 0.05
    :type threshold: float, optional
    :return: Trimmed experiment
    :rtype: oms.Experiment
    """
    tm = oms.ThresholdMower()
    params = tm.getDefaults()
    params.setValue("threshold", threshold)
    tm.setParameters(params)
    tm.filterPeakMap(experiment)
    return experiment

def trim_threshold_batch(experiments: Union[Sequence[Union[oms.MSExperiment,str]], str], run_dir:str,
                         file_ending:str=".mzML", threshold:float=0.05, deepcopy:bool=False) -> str:
    """
    Removes points below an absolute intensity theshold in a batch of experiments.

    :param experiments: Input Experiments (as a list of experiments, a directory, or paths to experiemnts)
    :type experiments: Union[Sequence[Union[pyopenms.MSExperiment,str]], str]
    :param threshold: Threshold for values to be excluded, defaults to 0.05
    :type threshold: float, optional
    :param deepcopy: Perform a deepcopy to relieably unlink the experiment from the input, defaults to False
    :type deepcopy: bool, optional
    :return: Folder with trimmed experiments
    :rtype: str
    """
    cleaned_dir = os.path.normpath( clean_dir(run_dir, "trimmed") )

    if deepcopy:
        experiments = [copy_experiment(experiment) for experiment in experiments]
    names = load_names_batch(experiments, file_ending)
    experiments = load_experiments(experiments, file_ending, data_load=False)
    for i, experiment in enumerate(tqdm(experiments)):
        experiment = load_experiment(experiment)
        trimmed_exp = trim_threshold(experiment, threshold)
        oms.MzMLFile().store(os.path.join(cleaned_dir, names[i] + ".mzML"), trimmed_exp)
        del trimmed_exp
    return cleaned_dir



# Combination
def sum_spectra(experiment:oms.MSExperiment) -> oms.MSSpectrum:
    """
    Sum up spectra in one experiment.

    :param experiment: Input experiment
    :type experiment: pyopenms.MSExperiment
    :return: Spectrum with summed intensities along m/z axis
    :rtype: pyopenms.MSSpectrum
    """
    spectrum = experiment.getSpectra()[0]
    intensities_all = np.zeros(spectrum.size())
    mzs_all = spectrum.get_peaks()[0]
    for spectrum in experiment.getSpectra():
        mzs, intensities = spectrum.get_peaks()
        intensities_all = np.sum([intensities_all, intensities], axis=0) # type: ignore
    comb_spectrum = oms.MSSpectrum()
    comb_spectrum.set_peaks( (mzs_all, intensities_all) ) # type: ignore
    return comb_spectrum


def combine_spectra_experiments(spectra_container:Sequence[Union[oms.MSExperiment,oms.MSSpectrum]]) -> oms.MSExperiment:
    """
    Combines all spectra/experiements, into different spectra in one experiment

    :param spectra_container: Input experiments
    :type spectra_container: Sequence[Union[oms.MSExperiment,oms.MSSpectrum]]
    :return: Experiment with summed intensities along m/z axis
    :rtype: oms.MSExperiment
    """
    experiment_all = oms.MSExperiment()
    for i, sc in enumerate(spectra_container):
        if isinstance(sc, oms.MSSpectrum):
            sc.setRT(float(i))
            experiment_all.addSpectrum(sc)
        else:
            for spectrum in sc.getSpectra():            # type: ignore
                spectrum.setRT(float(i))
                experiment_all.addSpectrum(spectrum)
    return experiment_all



# Smoothing
def smooth_spectra(experiment: Union[oms.MSExperiment, str], gaussian_width: float, deepcopy: bool = False) -> oms.MSExperiment:
    """
    Apply a Gaussian filter to all spectra in an experiment.

    :param experiment: Input experiment
    :type experiment: Union[pyopenms.MSExperiment, str]
    :param gaussian_width: Window width to apply smoothing to
    :type gaussian_width: float
    :param deepcopy: Perform a deepcopy to relieably unlink the experiment from the input, defaults to False
    :type deepcopy: bool, optional
    :return: Smoothed experiment
    :rtype: pyopenms.MSExperiment
    """
    experiment = load_experiment(experiment)

    if deepcopy:
        smooth_exp = copy_experiment(experiment)
    else:
        smooth_exp = experiment
    smooth_exp.setSpectra(experiment.getSpectra())
    gf = oms.GaussFilter()
    param = gf.getParameters()
    param.setValue("gaussian_width", gaussian_width)
    gf.setParameters(param)
    gf.filterExperiment(smooth_exp)
    return smooth_exp


# Centroiding
def centroid_experiment(experiment: Union[oms.MSExperiment, str], instrument:str="TOF",
                        signal_to_noise:float=1.0, spacing_difference_gap:float=4.0,
                        spacing_difference:float=1.5, missing:int=1, ms_levels:List[int]=[],
                        report_FWHM:str="true", report_FWHM_unit:str="relative", max_intensity:float=-1,
                        auto_max_stdev_factor:float=3.0, auto_max_percentile:int=95, auto_mode:int=0,
                        win_len:float=200.0, bin_count:int=30, min_required_elements:int=10, 
                        noise_for_empty_window:float=1e+20, write_log_messages:str="true",
                        peak_width:float=0.0, sn_bin_count:int=30, nr_iterations:int=5, sn_win_len:float=20.0,
                        check_width_internally:str="false", ms1_only:str="true", clear_meta_data:str="false",
                        deepcopy: bool = False) -> oms.MSExperiment:
    """
    Reduce dataset to centroids.

    Usecase
    fia_df["cent_experiment"] = fia_df["experiment"].apply(lambda experiment: centroid_experiment(experiment, instrument="TOF",                                      # For All
                                                                                                signal_to_noise=2.0, spacing_difference=1.5,
                                                                                                                        
                                                                                                spacing_difference_gap=4.0, missing=1, ms_levels=[1],                   # For Orbitrap
                                                                                                report_FWHM="true", report_FWHM_unit="relative", max_intensity=-1,
                                                                                                auto_max_stdev_factor=3.0, auto_max_percentile=95, auto_mode=0,
                                                                                                win_len=200.0, bin_count=30, min_required_elements=10, 
                                                                                                noise_for_empty_window=1e+20, write_log_messages="true",

                                                                                                peak_width=0.0, sn_bin_count=30, nr_iterations=5, sn_win_len=20.0,      # For TOF
                                                                                                check_width_internally="false", ms1_only="true", clear_meta_data="false",
                                                                                                deepcopy=False))

    :param experiment: Input experiment
    :type experiment: Union[pyopenms.MSExperiment, str]
    :param instrument: Instrument type (TOF or FT-ICR, Orbitrap), defaults to "TOF"
    :type instrument: str, optional
    :param signal_to_noise: Signal to noise ratio, defaults to 1.0
    :type signal_to_noise: float, optional
    :param spacing_difference_gap: Spacing difference gap, defaults to 4.0
    :type spacing_difference_gap: float, optional
    :param spacing_difference: Spacing difference, defaults to 1.5
    :type spacing_difference: float, optional
    :param missing: Number of allowed missing values, defaults to 1
    :type missing: int, optional
    :param ms_levels: MS levels to consider, defaults to []
    :type ms_levels: List[int], optional
    :param report_FWHM: Report full width at half maximum, defaults to "true"
    :type report_FWHM: str, optional
    :param report_FWHM_unit: Report full width at half maximum unit, defaults to "relative"
    :type report_FWHM_unit: str, optional
    :param max_intensity: Maximum intensity, defaults to -1
    :type max_intensity: float, optional
    :param auto_max_stdev_factor: Automatic maximal standard deviation factor, defaults to 3.0
    :type auto_max_stdev_factor: float, optional
    :param auto_max_percentile: Automatic maximal percentile to consider, defaults to 95
    :type auto_max_percentile: int, optional
    :param auto_mode: Automatic mode (0/1), defaults to 0
    :type auto_mode: int, optional
    :param win_len: Window length, defaults to 200.0
    :type win_len: float, optional
    :param bin_count: Number of bins, defaults to 30
    :type bin_count: int, optional
    :param min_required_elements: Minimum required elements for a peak, defaults to 10
    :type min_required_elements: int, optional
    :param noise_for_empty_window: Noise value for an empty window, defaults to 1e+20
    :type noise_for_empty_window: float, optional
    :param write_log_messages: Write log messages (true/false), defaults to "true"
    :type write_log_messages: str, optional
    :param peak_width: Expected peak width, defaults to 0.0
    :type peak_width: float, optional
    :param sn_bin_count: Signal bin count, defaults to 30
    :type sn_bin_count: int, optional
    :param nr_iterations: Iterations to recenter peaks, defaults to 5
    :type nr_iterations: int, optional
    :param sn_win_len: Signal window length, defaults to 20.0
    :type sn_win_len: float, optional
    :param check_width_internally: Check width internally, defaults to "false"
    :type check_width_internally: str, optional
    :param ms1_only: Only MS1 spectrum, defaults to "true"
    :type ms1_only: str, optional
    :param clear_meta_data: Clear meta data, defaults to "false"
    :type clear_meta_data: str, optional
    :param deepcopy: Perform a deepcopy to relieably unlink the experiment from the input, defaults to False
    :type deepcopy: bool, optional
    :return: Centroided experiment
    :rtype: pyopenms.MSExperiment
    """
    experiment = load_experiment(experiment)

    accu_exp = oms.MSExperiment()
    if instrument in ["FT-ICR-MS", "Orbitrap"]:
        pphr = oms.PeakPickerHiRes()
        params = pphr.getDefaults()
        params.setValue("signal_to_noise", signal_to_noise)
        params.setValue("spacing_difference_gap", spacing_difference_gap)
        params.setValue("spacing_difference", spacing_difference)
        params.setValue("missing", missing)
        params.setValue("ms_levels", ms_levels)
        params.setValue("report_FWHM", report_FWHM)
        params.setValue("report_FWHM_unit", report_FWHM_unit)
        params.setValue("SignalToNoise:max_intensity", max_intensity)
        params.setValue("SignalToNoise:auto_max_stdev_factor", auto_max_stdev_factor)
        params.setValue("SignalToNoise:auto_max_percentile", auto_max_percentile)
        params.setValue("SignalToNoise:auto_mode", auto_mode)
        params.setValue("SignalToNoise:win_len", win_len)
        params.setValue("SignalToNoise:bin_count", bin_count)
        params.setValue("SignalToNoise:min_required_elements", min_required_elements)
        params.setValue("SignalToNoise:noise_for_empty_window", noise_for_empty_window)
        params.setValue("SignalToNoise:write_log_messages", write_log_messages)
        pphr.pickExperiment(experiment, accu_exp, True)
    elif instrument in ["TOF-MS"]:
        ppi = oms.PeakPickerIterative()
        params = ppi.getDefaults()
        params.setValue("signal_to_noise", signal_to_noise)
        params.setValue("peak_width", peak_width)
        params.setValue("spacing_difference", spacing_difference)
        params.setValue("sn_bin_count_", sn_bin_count)
        params.setValue("nr_iterations_", nr_iterations)
        params.setValue("sn_win_len_", sn_win_len)
        params.setValue("check_width_internally", check_width_internally)
        params.setValue("ms1_only", ms1_only)
        params.setValue("clear_meta_data", clear_meta_data)
        ppi.pickExperiment(experiment, accu_exp)
    else:
        oms.PeakPickerHiRes().pickExperiment(experiment, accu_exp, True)

    centroid_exp = copy_experiment(experiment) if deepcopy else experiment
    centroid_exp.setSpectra(accu_exp.getSpectra())

    return centroid_exp

def centroid_batch(experiments: Union[Sequence[Union[oms.MSExperiment,str]], str], run_dir:str, file_ending:str=".mzML",
                   instrument:str="TOF",
                   signal_to_noise:float=1.0, spacing_difference_gap:float=4.0,
                   spacing_difference:float=1.5, missing:int=1, ms_levels:List[int]=[],
                   report_FWHM:str="true", report_FWHM_unit:str="relative", max_intensity:float=-1,
                   auto_max_stdev_factor:float=3.0, auto_max_percentile:int=95, auto_mode:int=0,
                   win_len:float=200.0, bin_count:int=30, min_required_elements:int=10,
                   noise_for_empty_window:float=1e+20, write_log_messages:str="true",
                   peak_width:float=0.0, sn_bin_count:int=30, nr_iterations:int=5, sn_win_len:float=20.0,
                   check_width_internally:str="false", ms1_only:str="true", clear_meta_data:str="false",
                   deepcopy:bool=False) -> str:
    """
    Centroids a batch of experiments, extracted from files in a given directory with a given file ending (i.e. .mzML or .mzXML).
    Returns the new directors as path/centroids.

    :param experiments: Input experiments
    :type experiments: Union[Sequence[Union[oms.MSExperiment,str]], str]
    :param run_dir: Run directory
    :type run_dir: str
    :param file_ending: File ending, defaults to ".mzML"
    :type file_ending: str, optional
    :param instrument: Instrument type (TOF or FT-ICR, Orbitrap), defaults to "TOF"
    :type instrument: str, optional
    :param signal_to_noise: Signal to noise ratio, defaults to 1.0
    :type signal_to_noise: float, optional
    :param spacing_difference_gap: Spacing difference gap, defaults to 4.0
    :type spacing_difference_gap: float, optional
    :param spacing_difference: Spacing difference, defaults to 1.5
    :type spacing_difference: float, optional
    :param missing: Number of allowed missing values, defaults to 1
    :type missing: int, optional
    :param ms_levels: MS levels to consider, defaults to []
    :type ms_levels: List[int], optional
    :param report_FWHM: Report full width at half maximum, defaults to "true"
    :type report_FWHM: str, optional
    :param report_FWHM_unit: Report full width at half maximum unit, defaults to "relative"
    :type report_FWHM_unit: str, optional
    :param max_intensity: Maximum intensity, defaults to -1
    :type max_intensity: float, optional
    :param auto_max_stdev_factor: Automatic maximal standard deviation factor, defaults to 3.0
    :type auto_max_stdev_factor: float, optional
    :param auto_max_percentile: Automatic maximal percentile to consider, defaults to 95
    :type auto_max_percentile: int, optional
    :param auto_mode: Automatic mode (0/1), defaults to 0
    :type auto_mode: int, optional
    :param win_len: Window length, defaults to 200.0
    :type win_len: float, optional
    :param bin_count: Number of bins, defaults to 30
    :type bin_count: int, optional
    :param min_required_elements: Minimum required elements for a peak, defaults to 10
    :type min_required_elements: int, optional
    :param noise_for_empty_window: Noise value for an empty window, defaults to 1e+20
    :type noise_for_empty_window: float, optional
    :param write_log_messages: Write log messages (true/false), defaults to "true"
    :type write_log_messages: str, optional
    :param peak_width: Expected peak width, defaults to 0.0
    :type peak_width: float, optional
    :param sn_bin_count: Signal bin count, defaults to 30
    :type sn_bin_count: int, optional
    :param nr_iterations: Iterations to recenter peaks, defaults to 5
    :type nr_iterations: int, optional
    :param sn_win_len: Signal window length, defaults to 20.0
    :type sn_win_len: float, optional
    :param check_width_internally: Check width internally, defaults to "false"
    :type check_width_internally: str, optional
    :param ms1_only: Only MS1 spectrum, defaults to "true"
    :type ms1_only: str, optional
    :param clear_meta_data: Clear meta data, defaults to "false"
    :type clear_meta_data: str, optional
    :param deepcopy: Perform a deepcopy to relieably unlink the experiment from the input, defaults to False
    :type deepcopy: bool, optional
    :return: Directory with centroids
    :rtype: str
    """
    cleaned_dir = os.path.normpath( clean_dir(run_dir, "centroids") )
    names = load_names_batch(experiments, file_ending)
    experiments = load_experiments(experiments, file_ending, data_load=False)
    for i, experiment in enumerate(tqdm(experiments)):
        experiment = load_experiment(experiment)
        centroided_exp = centroid_experiment(experiment,
                                            instrument=instrument,
                                            signal_to_noise=signal_to_noise, spacing_difference_gap=spacing_difference_gap,
                                            spacing_difference=spacing_difference, missing=missing, ms_levels=ms_levels,
                                            report_FWHM=report_FWHM, report_FWHM_unit=report_FWHM_unit, max_intensity=max_intensity,
                                            auto_max_stdev_factor=auto_max_stdev_factor, auto_max_percentile=auto_max_percentile,
                                            auto_mode=auto_mode, win_len=win_len, bin_count=bin_count,
                                            min_required_elements=min_required_elements, noise_for_empty_window=noise_for_empty_window,
                                            write_log_messages=write_log_messages,
                                            peak_width=peak_width, sn_bin_count=sn_bin_count,
                                            nr_iterations=nr_iterations, sn_win_len=sn_win_len,check_width_internally=check_width_internally,
                                            ms1_only=ms1_only, clear_meta_data=clear_meta_data,
                                            deepcopy=deepcopy)
        oms.MzMLFile().store(os.path.join(cleaned_dir, names[i] + ".mzML"), centroided_exp)
    return cleaned_dir



# Merging
def merge_experiment(experiment: Union[oms.MSExperiment, str], method:str="block_method",
                     mz_binning_width:float=1.0, mz_binning_width_unit:str="ppm", ms_levels:List[int]=[1], sort_blocks:str="RT_ascending",
                     rt_block_size: Optional[int] = None, rt_max_length:float=0.0,
                     spectrum_type:str="automatic", rt_range:Optional[float]=5.0, rt_unit:str="scans", 
                     rt_FWHM:float=5.0, cutoff:float=0.01, precursor_mass_tol:float=0.0, precursor_max_charge:int=1,
                     deepcopy: bool = False) -> oms.MSExperiment:
    """
    Merge several spectra into one spectrum (useful for MS1 spectra to amplify signals)

    :param experiment: Input experiment
    :type experiment: Union[oms.MSExperiment, str]
    :param method: Method to perform merging, defaults to "block_method"
    :type method: str, optional
    :param mz_binning_width: m/z binning width, defaults to 1.0
    :type mz_binning_width: float, optional
    :param mz_binning_width_unit: m/z binning width unit (ppm/Da), defaults to "ppm"
    :type mz_binning_width_unit: str, optional
    :param ms_levels: MS levels to consider, defaults to [1]
    :type ms_levels: List[int], optional
    :param sort_blocks: Sort blocks by rentention time, defaults to "RT_ascending"
    :type sort_blocks: str, optional
    :param rt_block_size: Block size along retention time, defaults to None
    :type rt_block_size: Optional[int], optional
    :param rt_max_length: Maximal length of Retention time, defaults to 0.0
    :type rt_max_length: float, optional
    :param spectrum_type: Spectrum type determination, defaults to "automatic"
    :type spectrum_type: str, optional
    :param rt_range: Retention time range to merge over, defaults to 5.0
    :type rt_range: Optional[float], optional
    :param rt_unit: Unit to merge over, defaults to "scans"
    :type rt_unit: str, optional
    :param rt_FWHM: Full width at half maximum, defaults to 5.0
    :type rt_FWHM: float, optional
    :param cutoff: Cutoff value during merging, defaults to 0.01
    :type cutoff: float, optional
    :param precursor_mass_tol: Percursor mass tolerance, defaults to 0.0
    :type precursor_mass_tol: float, optional
    :param precursor_max_charge: Maximal precursor charge, defaults to 1
    :type precursor_max_charge: int, optional
    :param deepcopy: Perform a deepcopy to relieably unlink the experiment from the input, defaults to False
    :type deepcopy: bool, optional
    :raises ValueError: Method needs to be in [block_method|average_tophat|average_gaussian]
    :return: Merged experiment
    :rtype: pyopenms.MSExperiment
    """
    experiment = load_experiment(experiment)

    if rt_block_size is None:
        rt_block_size = experiment.getNrSpectra()

    if deepcopy:
        merge_exp = copy_experiment(experiment)
    else:
        merge_exp = experiment

    merger = oms.SpectraMerger()
    param = merger.getParameters()
    param.setValue("mz_binning_width", mz_binning_width)
    param.setValue("mz_binning_width_unit", mz_binning_width_unit)
    param.setValue("sort_blocks", sort_blocks)
    if method == "block_method":
        param.setValue("block_method:ms_levels", ms_levels)
        param.setValue("block_method:rt_block_size", rt_block_size)
        param.setValue("block_method:rt_max_length", rt_max_length)
    elif method == "average_tophat":
        param.setValue("average_tophat:ms_level", ms_levels[-1])
        param.setValue("average_tophat:spectrum_type", spectrum_type)
        param.setValue("average_tophat:rt_unit", rt_unit)
        if rt_unit == "scans" and not rt_range:
            rt_range = float(experiment.getNrSpectra())
        param.setValue("average_tophat:rt_range", rt_range)
    elif method == "average_gaussian":
        param.setValue("average_gaussian:ms_level", ms_levels[-1])
        param.setValue("average_gaussian:spectrum_type", spectrum_type)
        param.setValue("average_gaussian:rt_FWHM", rt_FWHM)
        param.setValue("average_gaussian:cutoff", cutoff)
        param.setValue("average_gaussian:precursor_mass_tol", precursor_mass_tol)
        param.setValue("average_gaussian:precursor_max_charge", precursor_max_charge)
    else:
        raise ValueError(f"method needs to be one of block_method|average_tophat|average_gaussian")
    
    merger.setParameters(param)

    if method in ["average_tophat", "average_gaussian"]:
        merger.average(merge_exp, method.split("_")[-1])
    else:
        merger.mergeSpectraBlockWise(merge_exp)

    return merge_exp


def merge_batch(experiments: Union[Sequence[Union[oms.MSExperiment,str]], str], run_dir:str, file_ending:str=".mzML", method:str="block_method",
                mz_binning_width:float=1.0, mz_binning_width_unit:str="ppm", ms_levels:List[int]=[1], sort_blocks:str="RT_ascending",
                rt_block_size: Optional[int] = None, rt_max_length:float=0.0,
                spectrum_type:str="automatic", rt_range:Optional[float]=5.0, rt_unit:str="scans", 
                rt_FWHM:float=5.0, cutoff:float=0.01, precursor_mass_tol:float=0.0, precursor_max_charge:int=1,
                deepcopy: bool = False) -> str:
    """
    Merge several spectra into one spectrum (useful for MS1 spectra to amplify signals along near retention times)

    :param experiments: Input experiments
    :type experiments: Union[Sequence[Union[oms.MSExperiment,str]], str]
    :param run_dir: Run directory
    :type run_dir: str
    :param file_ending: File ending, defaults to ".mzML"
    :type file_ending: str, optional
    :param method: Method to perform merging, defaults to "block_method"
    :type method: str, optional
    :param mz_binning_width: m/z binning width, defaults to 1.0
    :type mz_binning_width: float, optional
    :param mz_binning_width_unit: m/z binning width unit (ppm/Da), defaults to "ppm"
    :type mz_binning_width_unit: str, optional
    :param ms_levels: MS levels to consider, defaults to [1]
    :type ms_levels: List[int], optional
    :param sort_blocks: Sort blocks by rentention time, defaults to "RT_ascending"
    :type sort_blocks: str, optional
    :param rt_block_size: Block size along retention time, defaults to None
    :type rt_block_size: Optional[int], optional
    :param rt_max_length: Maximal length of Retention time, defaults to 0.0
    :type rt_max_length: float, optional
    :param spectrum_type: Spectrum type determination, defaults to "automatic"
    :type spectrum_type: str, optional
    :param rt_range: Retention time range to merge over, defaults to 5.0
    :type rt_range: Optional[float], optional
    :param rt_unit: Unit to merge over, defaults to "scans"
    :type rt_unit: str, optional
    :param rt_FWHM: Full width at half maximum, defaults to 5.0
    :type rt_FWHM: float, optional
    :param cutoff: Cutoff value during merging, defaults to 0.01
    :type cutoff: float, optional
    :param precursor_mass_tol: Percursor mass tolerance, defaults to 0.0
    :type precursor_mass_tol: float, optional
    :param precursor_max_charge: Maximal precursor charge, defaults to 1
    :type precursor_max_charge: int, optional
    :param deepcopy: Perform a deepcopy to relieably unlink the experiment from the input, defaults to False
    :type deepcopy: bool, optional
    :raises ValueError: Method needs to be in [block_method|average_tophat|average_gaussian]
    :return: Directory with merged experiments
    :rtype: str
    """
    cleaned_dir = os.path.normpath( clean_dir(run_dir, "merged") )
    names = load_names_batch(experiments, file_ending)
    experiments = load_experiments(experiments, file_ending, data_load=False)
    for i, experiment in enumerate(tqdm(experiments)):
        experiment = load_experiment(experiment)
        merged_exp = merge_experiment(experiment, method=method,
                                        mz_binning_width=mz_binning_width, mz_binning_width_unit=mz_binning_width_unit,
                                        ms_levels=ms_levels, sort_blocks=sort_blocks,
                                        rt_block_size=rt_block_size, rt_max_length=rt_max_length,
                                        spectrum_type=spectrum_type, rt_range=rt_range, rt_unit=rt_unit,
                                        rt_FWHM=rt_FWHM, cutoff=cutoff, precursor_mass_tol=precursor_mass_tol, precursor_max_charge=precursor_max_charge,
                                        deepcopy=deepcopy)
        oms.MzMLFile().store(os.path.join(cleaned_dir, names[i] + ".mzML"), merged_exp)

    return cleaned_dir


def make_merge_dict(dir:str, file_ending:str=".mzML") -> dict:
    """
    Create a dictionary for merging.

    :param dir: Directory to extract samples from.
    :type dir: str
    :param file_ending: File ending, defaults to ".mzML"
    :type file_ending: str, optional
    :return: Dictonary with sample names as keys and paths for merging as values
    :rtype: dict
    """    
    names = load_names_batch(dir)
    samples = {"_".join(name.split("_")[:-1]) for name in names}
    return {sample: [os.path.join(dir, file) for file in os.listdir(dir) if file.startswith(sample) and file.endswith(file_ending)] for sample in samples}


def merge_experiments(experiments: Union[Sequence[Union[oms.MSExperiment,str]], str], run_dir:str, file_ending:str=".mzML", method:str="block_method",
                    mz_binning_width:float=1.0, mz_binning_width_unit:str="ppm", ms_levels:List[int]=[1], sort_blocks:str="RT_ascending",
                    rt_block_size: Optional[int] = None, rt_max_length:float=0.0,
                    spectrum_type:str="automatic", rt_range:Optional[float]=5.0, rt_unit:str="scans", 
                    rt_FWHM:float=5.0, cutoff:float=0.01, precursor_mass_tol:float=0.0, precursor_max_charge:int=1,
                    deepcopy: bool = False) -> oms.MSExperiment:
    """
    Merge several spectra into one spectrum (useful for MS1 spectra to amplify signals along near retention times).
    Combines all spectra of given experiments into one experiment to merge over.

    :param experiments: Input experiments
    :type experiments: Union[Sequence[Union[oms.MSExperiment,str]], str]
    :param run_dir: Run directory
    :type run_dir: str
    :param file_ending: File ending, defaults to ".mzML"
    :type file_ending: str, optional
    :param method: Method to perform merging, defaults to "block_method"
    :type method: str, optional
    :param mz_binning_width: m/z binning width, defaults to 1.0
    :type mz_binning_width: float, optional
    :param mz_binning_width_unit: m/z binning width unit (ppm/Da), defaults to "ppm"
    :type mz_binning_width_unit: str, optional
    :param ms_levels: MS levels to consider, defaults to [1]
    :type ms_levels: List[int], optional
    :param sort_blocks: Sort blocks by rentention time, defaults to "RT_ascending"
    :type sort_blocks: str, optional
    :param rt_block_size: Block size along retention time, defaults to None
    :type rt_block_size: Optional[int], optional
    :param rt_max_length: Maximal length of Retention time, defaults to 0.0
    :type rt_max_length: float, optional
    :param spectrum_type: Spectrum type determination, defaults to "automatic"
    :type spectrum_type: str, optional
    :param rt_range: Retention time range to merge over, defaults to 5.0
    :type rt_range: Optional[float], optional
    :param rt_unit: Unit to merge over, defaults to "scans"
    :type rt_unit: str, optional
    :param rt_FWHM: Full width at half maximum, defaults to 5.0
    :type rt_FWHM: float, optional
    :param cutoff: Cutoff value during merging, defaults to 0.01
    :type cutoff: float, optional
    :param precursor_mass_tol: Percursor mass tolerance, defaults to 0.0
    :type precursor_mass_tol: float, optional
    :param precursor_max_charge: Maximal precursor charge, defaults to 1
    :type precursor_max_charge: int, optional
    :param deepcopy: Perform a deepcopy to relieably unlink the experiment from the input, defaults to False
    :type deepcopy: bool, optional
    :raises ValueError: Method needs to be in [block_method|average_tophat|average_gaussian]
    :return: Directory with merged experiments
    :rtype: str
    """
    experiments = load_experiments(experiments, file_ending)
    experiment_all = combine_spectra_experiments(experiments)
    merged_exp = merge_experiment(experiment_all, method=method,
                                    mz_binning_width=mz_binning_width, mz_binning_width_unit=mz_binning_width_unit,
                                    ms_levels=ms_levels, sort_blocks=sort_blocks,
                                    rt_block_size=rt_block_size, rt_max_length=rt_max_length,
                                    spectrum_type=spectrum_type, rt_range=rt_range, rt_unit=rt_unit,
                                    rt_FWHM=rt_FWHM, cutoff=cutoff, precursor_mass_tol=precursor_mass_tol, precursor_max_charge=precursor_max_charge,
                                    deepcopy=deepcopy)
    
    return merged_exp


def merge_mz_tolerance(comb_df:pd.DataFrame, charge:int=1, tolerance:float=1e-3, binned:bool=False) -> pd.DataFrame:
    """
    Weighted average of m/z values that are within absolute tolerance of a row in the primary dataframe.

    :param comb_df: Dataframe for merging
    :type comb_df: pandas.DataFrame
    :param charge: Theoretical charge, defaults to 1
    :type charge: int, optional
    :param tolerance: Tolerance of m/z difference, defaults to 1e-3
    :type tolerance: float, optional
    :param binned: Values already binned (simpler process), defaults to False
    :type binned: bool, optional
    :return: Merged Dataframe
    :rtype: pandas.DataFrame
    """    
    df1 = comb_df.loc[comb_df["polarity"] == charge, "clustered_experiment"].item().get_df(long=True)[["mz", "inty"]]
    df2 = comb_df.loc[comb_df["polarity"] == -charge, "clustered_experiment"].item().get_df(long=True)[["mz", "inty"]]
    df_comb = pd.concat([df1, df2]).sort_values("mz").reset_index(drop=True)

    mzs = []
    intys = []
    if binned:
        bins = np.append(np.arange(df_comb["mz"].head(1).item(), df_comb["mz"].tail(1).item(), tolerance), df_comb["mz"].tail(1).item())
        digitized = np.digitize(df_comb["mz"], bins)

        for i in tqdm(np.unique(digitized)):
            mzs.append(np.average(df_comb.loc[digitized == i]["mz"], weights=df_comb.loc[digitized == i]["inty"]))
            intys.append(np.mean(df_comb.loc[digitized == i]["inty"]))
    else:
        with tqdm(total=len(df_comb)) as progress_bar:
            while not df_comb.empty:
                mz_comp = df_comb.at[df_comb.index[0], "mz"]
                inty_comp = df_comb.at[df_comb.index[0], "inty"]
                comb = np.isclose(mz_comp, df_comb["mz"], rtol=0.0, atol=tolerance)

                mzs.append(np.average(np.append(df_comb.loc[comb]["mz"], mz_comp), weights=np.append(df_comb.loc[comb]["inty"], inty_comp)))
                intys.append(np.average(np.append(df_comb.loc[comb]["inty"], mz_comp)))
                
                df_comb = df_comb.loc[~comb]
                progress_bar.update(sum(comb))

    merged_df = pd.DataFrame({"mz": mzs, "inty": intys})
    return merged_df.sort_values("mz").reset_index(drop=True)



# Normalization
def normalize_spectra(experiment: Union[oms.MSExperiment, str], normalization_method: str = "to_one",
                      deepcopy: bool = False) -> oms.MSExperiment:
    """
    Normalizes spectra by specified method.

    :param experiment: Input experiment
    :type experiment: Union[oms.MSExperiment, str]
    :param normalization_method: _description_, defaults to "to_one"
    :type normalization_method: Normalization method in [to_TIC | to_one] , optional
    :param deepcopy: Perform a deepcopy to relieably unlink the experiment from the input, defaults to False
    :type deepcopy: bool, optional
    :return: Normlaized experiment along spectra
    :rtype: oms.MSExperiment
    """
    experiment = load_experiment(experiment)

    if deepcopy:
        norm_exp = copy_experiment(experiment)
    else:
        norm_exp = experiment
    norm_exp.setSpectra(experiment.getSpectra())

    normalizer = oms.Normalizer()
    param = normalizer.getParameters()
    param.setValue("method", normalization_method)
    normalizer.setParameters(param)

    normalizer.filterPeakMap(norm_exp)
    return norm_exp

# Deisotoping
def deisotope_spectrum(spectrum: oms.MSSpectrum, fragment_tolerance: float = 0.1, fragment_unit_ppm: bool = False,
                       min_charge: int = 1, max_charge: int = 3,
                       keep_only_deisotoped: bool = True, min_isopeaks: int = 2, max_isopeaks: int = 10,
                       make_single_charged: bool = True, annotate_charge: bool = True,
                       annotate_iso_peak_count: bool = True, use_decreasing_model: bool = True,
                       start_intensity_check: bool = False, add_up_intensity: bool = False) -> oms.MSSpectrum:
    """
    Attempt to combine isotopes in a spectrum through exhaustive calculation.

    :param spectrum: Input spectrum
    :type spectrum: pyopenms.MSSpectrum
    :param fragment_tolerance: Tolerance for fragments, defaults to 0.1
    :type fragment_tolerance: float, optional
    :param fragment_unit_ppm: Use ppm as fragmentation unit, defaults to False
    :type fragment_unit_ppm: bool, optional
    :param min_charge: Minimal charge, defaults to 1
    :type min_charge: int, optional
    :param max_charge: Maximum charge, defaults to 3
    :type max_charge: int, optional
    :param keep_only_deisotoped: Keep only deisotoped signals, defaults to True
    :type keep_only_deisotoped: bool, optional
    :param min_isopeaks: Minimum amount of isotopes for a signal, defaults to 2
    :type min_isopeaks: int, optional
    :param max_isopeaks: Maximum amount of isotopes in a signal, defaults to 10
    :type max_isopeaks: int, optional
    :param make_single_charged: Adapt isotops to single hydrogen adducts/deducts, defaults to True
    :type make_single_charged: bool, optional
    :param annotate_charge: Annotate the charge, defaults to True
    :type annotate_charge: bool, optional
    :param annotate_iso_peak_count: Annotate isotopic peak count, defaults to True
    :type annotate_iso_peak_count: bool, optional
    :param use_decreasing_model: Use decreasing model (decreased chance of isotopes with further changes), defaults to True
    :type use_decreasing_model: bool, optional
    :param start_intensity_check: Intensity check at the start, defaults to False
    :type start_intensity_check: bool, optional
    :param add_up_intensity: Add intensity of isotopes, defaults to False
    :type add_up_intensity: bool, optional
    :return: Deisotoped spectrum
    :rtype: pyopenms.MSSpectrum
    """    
    spectrum.setFloatDataArrays([])

    oms.Deisotoper.deisotopeAndSingleCharge(
        spectra=spectrum,
        fragment_tolerance=fragment_tolerance,
        fragment_unit_ppm=fragment_unit_ppm,
        min_charge=min_charge,
        max_charge=max_charge,
        keep_only_deisotoped=keep_only_deisotoped,
        min_isopeaks=min_isopeaks,
        max_isopeaks=max_isopeaks,
        make_single_charged=make_single_charged,
        annotate_charge=annotate_charge,
        annotate_iso_peak_count=annotate_iso_peak_count,
        use_decreasing_model=use_decreasing_model,
        start_intensity_check=start_intensity_check,
        add_up_intensity=add_up_intensity
    )

    return spectrum


def deisotope_experiment(experiment: Union[oms.MSExperiment, str], fragment_tolerance: float = 0.1, fragment_unit_ppm: bool = False,
                         min_charge: int = 1, max_charge: int = 3,
                         keep_only_deisotoped: bool = True, min_isopeaks: int = 2, max_isopeaks: int = 10,
                         make_single_charged: bool = True, annotate_charge: bool = True,
                         annotate_iso_peak_count: bool = True, use_decreasing_model: bool = True,
                         start_intensity_check: bool = False, add_up_intensity: bool = False,
                         deepcopy: bool = False):
    """
    Attempt to combine isotopes in an experiment through exhaustive calculation.

    :param spectrum: Input experiment
    :type spectrum: Union[oms.MSExperiment, str]
    :param fragment_tolerance: Tolerance for fragments, defaults to 0.1
    :type fragment_tolerance: float, optional
    :param fragment_unit_ppm: Use ppm as fragmentation unit, defaults to False
    :type fragment_unit_ppm: bool, optional
    :param min_charge: Minimal charge, defaults to 1
    :type min_charge: int, optional
    :param max_charge: Maximum charge, defaults to 3
    :type max_charge: int, optional
    :param keep_only_deisotoped: Keep only deisotoped signals, defaults to True
    :type keep_only_deisotoped: bool, optional
    :param min_isopeaks: Minimum amount of isotopes for a signal, defaults to 2
    :type min_isopeaks: int, optional
    :param max_isopeaks: Maximum amount of isotopes in a signal, defaults to 10
    :type max_isopeaks: int, optional
    :param make_single_charged: Adapt isotops to single hydrogen adducts/deducts, defaults to True
    :type make_single_charged: bool, optional
    :param annotate_charge: Annotate the charge, defaults to True
    :type annotate_charge: bool, optional
    :param annotate_iso_peak_count: Annotate isotopic peak count, defaults to True
    :type annotate_iso_peak_count: bool, optional
    :param use_decreasing_model: Use decreasing model (decreased chance of isotopes with further changes), defaults to True
    :type use_decreasing_model: bool, optional
    :param start_intensity_check: Intensity check at the start, defaults to False
    :type start_intensity_check: bool, optional
    :param add_up_intensity: Add intensity of isotopes, defaults to False
    :type add_up_intensity: bool, optional
    :return: Deisotoped experiment
    :rtype: oms.MSExperiment
    """    

    experiment = load_experiment(experiment)

    if deepcopy:
        deisotop_exp = copy_experiment(experiment)
    else:
        deisotop_exp = experiment
    for i, spectrum in enumerate(deisotop_exp):
        deisotop_exp[i] = deisotope_spectrum(spectrum, fragment_tolerance, fragment_unit_ppm, min_charge, max_charge,
                                             keep_only_deisotoped,
                                             min_isopeaks, max_isopeaks, make_single_charged, annotate_charge,
                                             annotate_iso_peak_count,
                                             use_decreasing_model, start_intensity_check, add_up_intensity)
    return deisotop_exp



## Feature detection
def mass_trace_detection(experiment: Union[oms.MSExperiment, str],
                         mass_error_ppm: float = 10.0, noise_threshold_int: float = 1000.0, reestimate_mt_sd:str="true",
                         quant_method:str="median", trace_termination_criterion:str="outlier", trace_termination_outliers:int=3,
                         min_trace_length:float=5.0, max_trace_length:float=-1.0) -> list:
    """
    Detection of mass traces in experiment over several spectra.

    :param experiment: Input experiment
    :type experiment: Union[oms.MSExperiment, str]
    :param mass_error_ppm: Mass error in ppm, defaults to 10.0
    :type mass_error_ppm: float, optional
    :param noise_threshold_int: Noise threshold intensity, defaults to 1000.0
    :type noise_threshold_int: float, optional
    :param reestimate_mt_sd: Reestimate mass trace standard deviation during run, defaults to "true"
    :type reestimate_mt_sd: str, optional
    :param quant_method: Quantification method, defaults to "median"
    :type quant_method: str, optional
    :param trace_termination_criterion: Criterion to terminate a trace, defaults to "outlier"
    :type trace_termination_criterion: str, optional
    :param trace_termination_outliers: Number of cases that fulfil criterion/outliers to break trace, defaults to 3
    :type trace_termination_outliers: int, optional
    :param min_trace_length: Minimal trace length, defaults to 5.0
    :type min_trace_length: float, optional
    :param max_trace_length: Maximum trace length, defaults to -1.0
    :type max_trace_length: float, optional
    :return: Mass traces
    :rtype: list
    """
    experiment = load_experiment(experiment)
    experiment.sortSpectra(True)

    mass_traces = []
    mtd = oms.MassTraceDetection()
    mtd_par = mtd.getDefaults()
    mtd_par.setValue("mass_error_ppm", mass_error_ppm)
    mtd_par.setValue("noise_threshold_int", noise_threshold_int)
    mtd_par.setValue("reestimate_mt_sd", reestimate_mt_sd)              # Dynamic re-estimation of m/z variance
    mtd_par.setValue("quant_method", quant_method)                      # Method of quantification for mass traces. "median" is recommended for direct injection
    mtd_par.setValue("trace_termination_criterion", trace_termination_criterion)
    mtd_par.setValue("trace_termination_outliers", trace_termination_outliers) 
    mtd_par.setValue("min_trace_length", min_trace_length)
    mtd_par.setValue("max_trace_length", max_trace_length)
    mtd.setParameters(mtd_par)
    mtd.run(experiment, mass_traces, 0)

    return mass_traces

def mass_trace_detection_batch(experiments: Union[Sequence[Union[oms.MSExperiment,str]], str], file_ending:str=".mzML", 
                               mass_error_ppm: float = 10.0, noise_threshold_int: float = 1000.0, reestimate_mt_sd:str="true",
                               quant_method:str="median", trace_termination_criterion:str="outlier", trace_termination_outliers:int=3,
                               min_trace_length:float=5.0, max_trace_length:float=-1.0) -> list:
    """
    Detection of mass traces in experiment over several experiments.

    :param experiments: Input experiments
    :type experiments: Union[Sequence[Union[oms.MSExperiment,str]], str]
    :param file_ending: File ending, defaults to ".mzML"
    :type file_ending: str, optional
    :param mass_error_ppm: Mass error in ppm, defaults to 10.0
    :type mass_error_ppm: float, optional
    :param noise_threshold_int: Noise threshold intensity, defaults to 1000.0
    :type noise_threshold_int: float, optional
    :param reestimate_mt_sd: Reestimate mass trace standard deviation during run, defaults to "true"
    :type reestimate_mt_sd: str, optional
    :param quant_method: Quantification method, defaults to "median"
    :type quant_method: str, optional
    :param trace_termination_criterion: Criterion to terminate a trace, defaults to "outlier"
    :type trace_termination_criterion: str, optional
    :param trace_termination_outliers: Number of cases that fulfil criterion/outliers to break trace, defaults to 3
    :type trace_termination_outliers: int, optional
    :param min_trace_length: Minimal trace length, defaults to 5.0
    :type min_trace_length: float, optional
    :param max_trace_length: Maximum trace length, defaults to -1.0
    :type max_trace_length: float, optional
    :return: Mass traces
    :rtype: list
    """    
    mass_traces_all = []
    experiments = load_experiments(experiments, file_ending, data_load=False)
    for experiment in tqdm(experiments):
        experiment = load_experiment(experiment)
        mass_traces_all.append(
            mass_trace_detection(experiment=experiment, mass_error_ppm=mass_error_ppm, noise_threshold_int=noise_threshold_int,
                                reestimate_mt_sd=reestimate_mt_sd, quant_method=quant_method, trace_termination_criterion=trace_termination_criterion,
                                trace_termination_outliers=trace_termination_outliers, min_trace_length=min_trace_length, max_trace_length=max_trace_length)
        )
            
    return mass_traces_all


def elution_peak_detection(mass_traces: list, chrom_fwhm:float=10.0, chrom_peak_snr:float=2.0,
                           width_filtering: str = "fixed", min_fwhm:float=1.0, max_fwhm:float=60.0,
                           masstrace_snr_filtering:str="false") -> list:
    """
    Elution peak detection along mass traces. Relevant for chromatographic data.

    :param mass_traces: List of mass traces
    :type mass_traces: list
    :param chrom_fwhm: Chromatographic full width at half maximum, defaults to 10.0
    :type chrom_fwhm: float, optional
    :param chrom_peak_snr: Minimum signal-to-noise a mass trace should have, defaults to 2.0
    :type chrom_peak_snr: float, optional
    :param width_filtering: Type of width filtering, defaults to "fixed"
    :type width_filtering: str, optional
    :param min_fwhm: Minimal full width at half maximum, defaults to 1.0
    :type min_fwhm: float, optional
    :param max_fwhm: Maximal full width at half maximum, defaults to 60.0
    :type max_fwhm: float, optional
    :param masstrace_snr_filtering: Filtering by signal to noise ratio, defaults to "false"
    :type masstrace_snr_filtering: str, optional
    :return: List of final mass traces
    :rtype: list
    """
    mass_traces_deconvol = []
    epd = oms.ElutionPeakDetection()
    epd_par = epd.getDefaults()
    # The fixed setting filters out mass traces outside the [min_fwhm: 1.0, max_fwhm: 60.0] interval
    epd_par.setValue("chrom_fwhm", chrom_fwhm)              # full-width-at-half-maximum of chromatographic peaks (s)
    epd_par.setValue("chrom_peak_snr", chrom_peak_snr)     # Signal to noise minimum
    epd_par.setValue("width_filtering", width_filtering)    # auto=excludes 5% on edges, fixed: uses min_fwhm and max_fwhm
    epd_par.setValue("min_fwhm", min_fwhm)           # ignored when width_filtering="auto"
    epd_par.setValue("max_fwhm", max_fwhm)           # ignored when width_filtering="auto"
    epd_par.setValue("masstrace_snr_filtering", masstrace_snr_filtering)
    epd.setParameters(epd_par)
    epd.detectPeaks(mass_traces, mass_traces_deconvol)
    if epd.getParameters().getValue("width_filtering") == "auto":
        mass_traces_final = []
        epd.filterByPeakWidth(mass_traces_deconvol, mass_traces_final)
    else:
        mass_traces_final = mass_traces_deconvol

    return mass_traces_final

def elution_peak_detection_batch(mass_traces_all: list[list], chrom_fwhm:float=10.0, chrom_peak_snr:float=2.0,
                                 width_filtering: str = "fixed", min_fwhm:float=1.0, max_fwhm:float=60.0,
                                 masstrace_snr_filtering:str="false") -> list[list]:
    """
    Elution peak detection along list of mass traces. Relevant for chromatographic data.

    :param mass_traces_all: List of list of all mass traces
    :type mass_traces_all: list[list]
    :param chrom_fwhm: Chromatographic full width at half maximum, defaults to 10.0
    :type chrom_fwhm: float, optional
    :param chrom_peak_snr: Minimum signal-to-noise a mass trace should have, defaults to 2.0
    :type chrom_peak_snr: float, optional
    :param width_filtering: Type of width filtering, defaults to "fixed"
    :type width_filtering: str, optional
    :param min_fwhm: Minimal full width at half maximum, defaults to 1.0
    :type min_fwhm: float, optional
    :param max_fwhm: Maximal full width at half maximum, defaults to 60.0
    :type max_fwhm: float, optional
    :param masstrace_snr_filtering: Filtering by signal to noise ratio, defaults to "false"
    :type masstrace_snr_filtering: str, optional
    :return: List of list of final mass traces
    :rtype: list[list]
    """    
    mass_traces_all_final = []
    for mass_traces in tqdm(mass_traces_all):
        mass_traces_all_final.append(
            elution_peak_detection(mass_traces=mass_traces, chrom_fwhm=chrom_fwhm, chrom_peak_snr=chrom_peak_snr,
                                   width_filtering=width_filtering, min_fwhm=min_fwhm, max_fwhm=max_fwhm,
                                   masstrace_snr_filtering=masstrace_snr_filtering)
        )

    return mass_traces_all_final


def feature_detection_untargeted(experiment: Union[oms.MSExperiment, str],
                                 mass_traces_deconvol: list = [], isotope_filtering_model="metabolites (2% RMS)",
                                 local_rt_range:float=3.0, local_mz_range:float=5.0, 
                                 charge_lower_bound:int=1, charge_upper_bound:int=3,
                                 chrom_fwhm:float=10.0, report_summed_ints:str="true",
                                 enable_RT_filtering:str="false", mz_scoring_13C:str="false",
                                 use_smoothed_intensities:str="false", report_convex_hulls: str = "true",
                                 report_chromatograms:str="false", remove_single_traces: str = "true",
                                 mz_scoring_by_elements: str = "false", elements:str="CHNOPS") -> oms.FeatureMap:
    """
    Untargeted feature detection in an experiment.

    :param experiment: Input experiment
    :type experiment: Union[oms.MSExperiment, str]
    :param mass_traces_deconvol: Deconvoluted mass traces, defaults to []
    :type mass_traces_deconvol: list, optional
    :param isotope_filtering_model: Isotope filtering model, defaults to "metabolites (2% RMS)"
    :type isotope_filtering_model: str, optional
    :param local_rt_range: Local retention time range, defaults to 3.0
    :type local_rt_range: float, optional
    :param local_mz_range: Local m/z range, defaults to 5.0
    :type local_mz_range: float, optional
    :param charge_lower_bound: Lower charge bound, defaults to 1
    :type charge_lower_bound: int, optional
    :param charge_upper_bound: Upper charge bound, defaults to 3
    :type charge_upper_bound: int, optional
    :param chrom_fwhm: Chromatographic full width at half maximum, defaults to 10.0
    :type chrom_fwhm: float, optional
    :param report_summed_ints: Report summed intensities, defaults to "true"
    :type report_summed_ints: str, optional
    :param enable_RT_filtering: Enable retention time filtering, defaults to "false"
    :type enable_RT_filtering: str, optional
    :param mz_scoring_13C: Score m/z by looking at expected Carbon13 peaks, defaults to "false"
    :type mz_scoring_13C: str, optional
    :param use_smoothed_intensities: Use smoothed intensities (if smoothed before), defaults to "false"
    :type use_smoothed_intensities: str, optional
    :param report_convex_hulls: Report convex hulls, defaults to "true"
    :type report_convex_hulls: str, optional
    :param report_chromatograms: Report chromatograms, defaults to "false"
    :type report_chromatograms: str, optional
    :param remove_single_traces: Remove single traces (only appear at one retention time), defaults to "true"
    :type remove_single_traces: str, optional
    :param mz_scoring_by_elements: Score m/z by present elements, defaults to "false"
    :type mz_scoring_by_elements: str, optional
    :param elements: Elements to consider, defaults to "CHNOPS"
    :type elements: str, optional
    :return: Feature Map
    :rtype: pyopenms.FeatureMap
    """
    feature_map = oms.FeatureMap()  # output features
    chrom_out = []  # output chromatograms
    ffm = oms.FeatureFindingMetabo()

    if isinstance(experiment, str):
        name = experiment.encode()
    
    experiment = load_experiment(experiment)

    ffm_par = ffm.getDefaults()
    ffm_par.setValue("local_rt_range", local_rt_range)          # rt range for coeluting mass traces (can be set low (3.0s ~ 2 frames/spectra), because only one peak is expected)
    ffm_par.setValue("local_mz_range", local_mz_range)          # mz range for isotopic traces
    ffm_par.setValue("charge_lower_bound", charge_lower_bound)
    ffm_par.setValue("charge_upper_bound", charge_upper_bound)
    ffm_par.setValue("chrom_fwhm", chrom_fwhm)                  # Set expected chromatographic width according to elution detection parameter
    ffm_par.setValue("report_summed_ints", report_summed_ints)  # Sum intesity over all traces or use monoisotopic peak intensity ? (amplyfies signal with detected isotopes)
    ffm_par.setValue("enable_RT_filtering", enable_RT_filtering) # Require RT overlap. 'false' for direct injection
    ffm_par.setValue("isotope_filtering_model", isotope_filtering_model) # metabolites (2% RMS) = Support Vector Machine, with Root mean square deviation of 2% (for precise machines)
    ffm_par.setValue("mz_scoring_13C", mz_scoring_13C)  # Disable for general metabolomics
    ffm_par.setValue("use_smoothed_intensities", use_smoothed_intensities)  # Use Locally Weighted Scatterplot Smoothed intensities (useful, if intensity is mass-dependent (Orbitraps)) ?
    ffm_par.setValue("report_convex_hulls", report_convex_hulls)
    ffm_par.setValue("report_chromatograms", report_chromatograms)  # 'false', was not performed in Flow-injection
    ffm_par.setValue("remove_single_traces", remove_single_traces)  # 'false', there will be valuable single traces, because we have long traces, that may not match
    ffm_par.setValue("mz_scoring_by_elements", mz_scoring_by_elements) # 'true' to use expected element peaks to detect isotopes 
    ffm_par.setValue("elements", elements) # Elements, that are present in sample: "CHNOPS"  
    ffm.setParameters(ffm_par)

    ffm.run(mass_traces_deconvol, feature_map, chrom_out)
    feature_map.setUniqueIds()  # Assigns a new, valid unique id per feature
    feature_map.setPrimaryMSRunPath([name])

    return feature_map

def feature_detection_untargeted_batch(experiments:Union[Sequence[Union[oms.MSExperiment,str]], str], file_ending:str=".mzML",
                                       mass_traces_deconvol_all: list[list] = [], isotope_filtering_model="metabolites (2% RMS)",
                                       local_rt_range:float=3.0, local_mz_range:float=5.0, 
                                       charge_lower_bound:int=1, charge_upper_bound:int=3,
                                       chrom_fwhm:float=10.0, report_summed_ints:str="true",
                                       enable_RT_filtering:str="false", mz_scoring_13C:str="false",
                                       use_smoothed_intensities:str="false", report_convex_hulls: str = "true",
                                       report_chromatograms:str="false", remove_single_traces: str = "true",
                                       mz_scoring_by_elements: str = "false", elements:str="CHNOPS") -> list[oms.FeatureMap]:
    """
    Untargeted feature detection in experiments.

    :param experiments: Input experiments
    :type experiments: Union[Sequence[Union[oms.MSExperiment,str]], str]
    :param file_ending: File ending, defaults to ".mzML"
    :type file_ending: str, optional
    :param mass_traces_deconvol: Deconvoluted mass traces, defaults to []
    :type mass_traces_deconvol: list, optional
    :param isotope_filtering_model: Isotope filtering model, defaults to "metabolites (2% RMS)"
    :type isotope_filtering_model: str, optional
    :param local_rt_range: Local retention time range, defaults to 3.0
    :type local_rt_range: float, optional
    :param local_mz_range: Local m/z range, defaults to 5.0
    :type local_mz_range: float, optional
    :param charge_lower_bound: Lower charge bound, defaults to 1
    :type charge_lower_bound: int, optional
    :param charge_upper_bound: Upper charge bound, defaults to 3
    :type charge_upper_bound: int, optional
    :param chrom_fwhm: Chromatographic full width at half maximum, defaults to 10.0
    :type chrom_fwhm: float, optional
    :param report_summed_ints: Report summed intensities, defaults to "true"
    :type report_summed_ints: str, optional
    :param enable_RT_filtering: Enable retention time filtering, defaults to "false"
    :type enable_RT_filtering: str, optional
    :param mz_scoring_13C: Score m/z by looking at expected Carbon13 peaks, defaults to "false"
    :type mz_scoring_13C: str, optional
    :param use_smoothed_intensities: Use smoothed intensities (if smoothed before), defaults to "false"
    :type use_smoothed_intensities: str, optional
    :param report_convex_hulls: Report convex hulls, defaults to "true"
    :type report_convex_hulls: str, optional
    :param report_chromatograms: Report chromatograms, defaults to "false"
    :type report_chromatograms: str, optional
    :param remove_single_traces: Remove single traces (only appear at one retention time), defaults to "true"
    :type remove_single_traces: str, optional
    :param mz_scoring_by_elements: Score m/z by present elements, defaults to "false"
    :type mz_scoring_by_elements: str, optional
    :param elements: Elements to consider, defaults to "CHNOPS"
    :type elements: str, optional
    :return: List of Feature Maps
    :rtype: list[pyopenms.FeatureMap]
    """    
    feature_maps = []
    experiments = load_experiments(experiments, file_ending, data_load=False)
    for i, experiment in enumerate(tqdm(experiments)):
        experiment = load_experiment(experiment)
        feature_maps.append(
            feature_detection_untargeted(experiment=experiment,
                                         mass_traces_deconvol=mass_traces_deconvol_all[i], 
                                         isotope_filtering_model=isotope_filtering_model,
                                         local_rt_range=local_rt_range, local_mz_range=local_mz_range, 
                                         charge_lower_bound=charge_lower_bound, charge_upper_bound=charge_upper_bound,
                                         chrom_fwhm=chrom_fwhm, report_summed_ints=report_summed_ints,
                                         enable_RT_filtering=enable_RT_filtering, mz_scoring_13C=mz_scoring_13C,
                                         use_smoothed_intensities=use_smoothed_intensities, report_convex_hulls=report_convex_hulls,
                                         report_chromatograms=report_chromatograms, remove_single_traces=remove_single_traces,
                                         mz_scoring_by_elements=mz_scoring_by_elements, elements=elements)
        )

    return feature_maps


def assign_feature_maps_polarity(feature_maps:list, scan_polarity:Optional[str]=None) -> list:
    """
    Assigns the polarity to a list of feature maps, depending on "pos"/"neg" in file name.

    :param feature_maps: List of feature maps
    :type feature_maps: list
    :param scan_polarity: Scan polarity, defaults to None
    :type scan_polarity: Optional[str], optional
    :return: List of feature maps with annotated polarity
    :rtype: list
    """
    print("Assign polarity to feature maps:")
    for fm in tqdm(feature_maps):
        if scan_polarity:
            fm.setMetaValue("scan_polarity", scan_polarity)
        elif b"neg" in os.path.basename(fm.getMetaValue("spectra_data")[0]):
            fm.setMetaValue("scan_polarity", "negative")
        elif b"pos" in os.path.basename(fm.getMetaValue("spectra_data")[0]):
            fm.setMetaValue("scan_polarity", "positive")
        for feature in fm:
            if scan_polarity:
                feature.setMetaValue("scan_polarity", scan_polarity)
            elif b"neg" in os.path.basename(fm.getMetaValue("spectra_data")[0]):
                feature.setMetaValue("scan_polarity", "negative")
            elif b"pos" in os.path.basename(fm.getMetaValue("spectra_data")[0]):
                feature.setMetaValue("scan_polarity", "positive")
    return feature_maps


def detect_adducts(feature_maps: list, potential_adducts:Union[str, bytes]="[]", q_try:str="feature", mass_max_diff:float=10.0, unit:str="ppm", max_minority_bound:int=3,
                   verbose_level:int=0) -> list:
    """
    Attempt adduct detection through exhaustive calculations.

    :param feature_maps: List of feature maps
    :type feature_maps: list
    :param potential_adducts: Potential adducts to consider, defaults to "[]"
    :type potential_adducts: Union[str, bytes], optional
    :param q_try: Charge discovery dimension, defaults to "feature"
    :type q_try: str, optional
    :param mass_max_diff: Maximum mass difference, defaults to 10.0
    :type mass_max_diff: float, optional
    :param unit: Unit of mass difference, defaults to "ppm"
    :type unit: str, optional
    :param max_minority_bound: Maximum minority bound, defaults to 3
    :type max_minority_bound: int, optional
    :param verbose_level: Verbosity level, defaults to 0
    :type verbose_level: int, optional
    :return: Feature maps with removed adducts (deconvoluted)
    :rtype: list
    """
    feature_maps_adducts = []
    print("Detecting adducts:")
    for feature_map in tqdm(feature_maps):
        mfd = oms.MetaboliteFeatureDeconvolution()
        mdf_par = mfd.getDefaults()
        if potential_adducts:
            mdf_par.setValue("potential_adducts", potential_adducts)
        mdf_par.setValue("verbose_level", verbose_level)
        mdf_par.setValue("mass_max_diff", mass_max_diff)
        mdf_par.setValue("unit", unit)
        mdf_par.setValue("q_try", q_try)
        mdf_par.setValue("max_minority_bound", max_minority_bound)
        mfd.setParameters(mdf_par)
        feature_map_adduct = oms.FeatureMap()
        mfd.compute(feature_map, feature_map_adduct, oms.ConsensusMap(), oms.ConsensusMap())
        feature_maps_adducts.append(feature_map_adduct)

    return feature_maps_adducts


def align_retention_times(feature_maps: list, max_num_peaks_considered:int=-1,max_mz_difference:float=10.0, mz_unit:str="ppm",
                          superimposer_max_scaling:float=2.0 ) -> list:
    """
    Use as reference for alignment, the file with the largest number of features
    Works well if you have a pooled QC for example. Returns the aligned map at the first position.

    :param feature_maps: List of feature maps
    :type feature_maps: list
    :param max_num_peaks_considered: Maximum number of considered peaks, defaults to -1
    :type max_num_peaks_considered: int, optional
    :param max_mz_difference: Maximum m/z difference, defaults to 10.0
    :type max_mz_difference: float, optional
    :param mz_unit: Unit for m/z values, defaults to "ppm"
    :type mz_unit: str, optional
    :param superimposer_max_scaling: Maximum scaling during superimposition, defaults to 2.0
    :type superimposer_max_scaling: float, optional
    :return: List of feature maps with aligned retention times
    :rtype: list
    """
    print("Searching feature map with larges number of features:")
    ref_index = np.argmax([fm.size() for fm in feature_maps])
    feature_maps.insert(0, feature_maps.pop(ref_index))


    aligner = oms.MapAlignmentAlgorithmPoseClustering()

    trafos = {}

    # parameter optimization
    aligner_par = aligner.getDefaults()
    aligner_par.setValue( "max_num_peaks_considered", max_num_peaks_considered )  # -1 = infinite
    aligner_par.setValue( "pairfinder:distance_MZ:max_difference", max_mz_difference )
    aligner_par.setValue( "pairfinder:distance_MZ:unit", "ppm" )
    aligner_par.setValue( "superimposer:max_scaling", superimposer_max_scaling )
    
    aligner.setParameters(aligner_par)
    aligner.setReference(feature_maps[0])

    print("Aligning retention times:")
    for feature_map in tqdm(feature_maps[1:]):
        trafo = oms.TransformationDescription()  # save the transformed data points
        aligner.align(feature_map, trafo)
        trafos[feature_map.getMetaValue("spectra_data")[0].decode()] = trafo
        transformer = oms.MapAlignmentTransformer()
        transformer.transformRetentionTimes(feature_map, trafo, True)

    return feature_maps


def separate_feature_maps_pos_neg(feature_maps:list) -> list:
    """
    Separate the feature maps into positively and negatively charged feature maps.

    :param feature_maps: Input list of feature maps
    :type feature_maps: list
    :return: List of list of feature maps, separates into positive and negatively charged
    :rtype: list
    """    
    positive_features = []
    negative_features = []
    print("Separating feature maps:")
    for fm in tqdm(feature_maps):
        if fm.getMetaValue("scan_polarity") == "positive":
            positive_features.append(fm)
        elif fm.getMetaValue("scan_polarity") == "negative":
            negative_features.append(fm)
    return [positive_features, negative_features]


def consensus_features_linking(feature_maps: list, feature_grouper_type:str="QT") -> oms.ConsensusMap:
    """
    Linking features by consensus voting.

    :param feature_maps: List of feature maps
    :type feature_maps: list
    :param feature_grouper_type: Quality threshold clustering (QT) or k-dimensional tree clustering, defaults to "QT"
    :type feature_grouper_type: str, optional
    :raises ValueError: Use QT or KD for feature groupers.
    :return: Consensums map of features
    :rtype: pyopenms.ConsensusMap
    """    
    if feature_grouper_type == "KD":
        feature_grouper = oms.FeatureGroupingAlgorithmKD()
    elif feature_grouper_type == "QT":
        feature_grouper = oms.FeatureGroupingAlgorithmQT()
    else:
        raise ValueError(f"{feature_grouper_type} is not in list of implemented feature groupers. Choose from ['KD','QT'].")

    consensus_map = oms.ConsensusMap()
    file_descriptions = consensus_map.getColumnHeaders()

    for i, feature_map in enumerate(feature_maps):
        file_description = file_descriptions.get(i, oms.ColumnHeader())
        file_description.filename = os.path.basename(feature_map.getMetaValue("spectra_data")[0].decode())
        file_description.size = feature_map.size()
        file_description.unique_id = feature_map.getUniqueId()

        file_descriptions[i] = file_description

    feature_grouper.group(feature_maps, consensus_map)
    consensus_map.setColumnHeaders(file_descriptions)

    return consensus_map
    

# Untargeted
def untargeted_feature_detection(experiment: Union[oms.MSExperiment, str],
                                 feature_filepath: Optional[str] = None,
                                 mass_error_ppm: float = 5.0,
                                 noise_threshold_int: float = 3000.0,
                                 charge_lower_bound:int=1,
                                 charge_upper_bound:int=3,
                                 width_filtering: str = "fixed",
                                 isotope_filtering_model="none",
                                 remove_single_traces="true",
                                 mz_scoring_by_elements="false",
                                 report_convex_hulls="true",
                                 deepcopy: bool = False) -> oms.FeatureMap:
    """
    Untargeted detection of features. Combines mMass trace detection, elution peak detection and 
    feature finding.

    :param experiment: Input experiment
    :type experiment: Union[oms.MSExperiment, str]
    :param feature_filepath: Path to featureXML file, defaults to None
    :type feature_filepath: Optional[str], optional
    :param mass_error_ppm: Mass error in ppm, defaults to 10.0
    :type mass_error_ppm: float, optional
    :param noise_threshold_int: Noise threshold intensity, defaults to 1000.0
    :type noise_threshold_int: float, optional
    :param charge_lower_bound: Lower charge bound, defaults to 1
    :type charge_lower_bound: int, optional
    :param charge_upper_bound: Upper charge bound, defaults to 3
    :type charge_upper_bound: int, optional
    :param width_filtering: Type of width filtering, defaults to "fixed"
    :type width_filtering: str, optional
    :param isotope_filtering_model: Isotope filtering model, defaults to "metabolites (2% RMS)"
    :type isotope_filtering_model: str, optional
    :param remove_single_traces: Remove single traces, defaults to "true"
    :type remove_single_traces: str, optional
    :param mz_scoring_by_elements: Score m/z by present elements, defaults to "false"
    :type mz_scoring_by_elements: str, optional
    :param report_convex_hulls: Report convex hulls, defaults to "true"
    :type report_convex_hulls: str, optional
    :param deepcopy: Perform a deepcopy to relieably unlink the experiment from the input, defaults to False
    :type deepcopy: bool, optional
    :return: Feature map
    :rtype: pyopenms.FeatureMap
    """
    experiment = load_experiment(experiment)
    experiment.sortSpectra(True)

    # Mass trace detection
    mass_traces = mass_trace_detection(experiment, mass_error_ppm, noise_threshold_int)
     
    # Elution Peak Detection
    mass_traces_deconvol = elution_peak_detection(mass_traces, width_filtering=width_filtering)
    
    # Feature finding
    feature_map = feature_detection_untargeted(experiment=experiment,
                                               mass_traces_deconvol=mass_traces_deconvol,
                                               isotope_filtering_model=isotope_filtering_model,
                                               charge_lower_bound=charge_lower_bound,
                                               charge_upper_bound=charge_upper_bound,
                                               remove_single_traces=remove_single_traces, 
                                               mz_scoring_by_elements=mz_scoring_by_elements, 
                                               report_convex_hulls=report_convex_hulls)

    if feature_filepath:
        oms.FeatureXMLFile().store(feature_filepath, feature_map)

    return feature_map

def untargeted_features_detection(in_dir: str, run_dir:str, file_ending:str=".mzML",
                                    mass_error_ppm:float=10.0,
                                    noise_threshold_int:float=1000.0,
                                    charge_lower_bound:int=1,
                                    charge_upper_bound:int=3,
                                    width_filtering:str="fixed",
                                    isotope_filtering_model:str="none",
                                    remove_single_traces:str="true",
                                    mz_scoring_by_elements:str="false",
                                    report_convex_hulls:str="true",
                                    deepcopy:bool=False) -> list:
    """
    Untargeted detection of features. Combines mMass trace detection, elution peak detection and 
    feature finding.

    :param in_dir: Input directory
    :type in_dir: str
    :param run_dir: Run directory
    :type run_dir: str
    :param file_ending: File ending, defaults to ".mzML"
    :type file_ending: str, optional
    :param mass_error_ppm: Mass error in ppm, defaults to 10.0
    :type mass_error_ppm: float, optional
    :param noise_threshold_int: Noise threshold intensity, defaults to 1000.0
    :type noise_threshold_int: float, optional
    :param charge_lower_bound: Lower charge bound, defaults to 1
    :type charge_lower_bound: int, optional
    :param charge_upper_bound: Upper charge bound, defaults to 3
    :type charge_upper_bound: int, optional
    :param width_filtering: Type of width filtering, defaults to "fixed"
    :type width_filtering: str, optional
    :param isotope_filtering_model: Isotope filtering model, defaults to "metabolites (2% RMS)"
    :type isotope_filtering_model: str, optional
    :param remove_single_traces: Remove single traces, defaults to "true"
    :type remove_single_traces: str, optional
    :param mz_scoring_by_elements: Score m/z by present elements, defaults to "false"
    :type mz_scoring_by_elements: str, optional
    :param report_convex_hulls: Report convex hulls, defaults to "true"
    :type report_convex_hulls: str, optional
    :param deepcopy: Perform a deepcopy to relieably unlink the experiment from the input, defaults to False
    :type deepcopy: bool, optional
    :return: List of feature maps
    :rtype: list
    """
    feature_maps = []
    feature_folder = clean_dir(run_dir, "features")

    for file in tqdm(os.listdir(in_dir)):
        if file.endswith(file_ending):
            experiment_file = os.path.join(in_dir, file)
            feature_file = os.path.join(feature_folder, f"{file[:-len(file_ending)]}.featureXML")
            feature_map = untargeted_feature_detection(experiment=experiment_file,
                                                        feature_filepath=feature_file,
                                                        mass_error_ppm=mass_error_ppm, noise_threshold_int=noise_threshold_int,
                                                        charge_lower_bound=charge_lower_bound, charge_upper_bound=charge_upper_bound, 
                                                        width_filtering=width_filtering, isotope_filtering_model=isotope_filtering_model,
                                                        remove_single_traces=remove_single_traces, mz_scoring_by_elements=mz_scoring_by_elements,
                                                        report_convex_hulls=report_convex_hulls,
                                                        deepcopy=deepcopy)
            feature_maps.append(feature_map)

    return feature_maps

        

## Targeted
def feature_detection_targeted(experiment: Union[oms.MSExperiment, str], metab_table:list, feature_filepath:Optional[str]=None,
                               mz_window:float=5.0, rt_window:Optional[float]=None, n_isotopes:int=2, isotope_pmin:float=0.01,
                               peak_width:float=60.0) -> oms.FeatureMap:
    """
    Feature detection with a given metabolic table.

    :param experiment: Input experiment
    :type experiment: Union[oms.MSExperiment, str]
    :param metab_table: Metabilites table
    :type metab_table: list
    :param feature_filepath: Filpath to featureXML, defaults to None
    :type feature_filepath: Optional[str], optional
    :param mz_window: m/z window width, defaults to 5.0
    :type mz_window: float, optional
    :param rt_window: Retention time window width, defaults to None
    :type rt_window: Optional[float], optional
    :param n_isotopes: Number of considered isotopes, defaults to 2
    :type n_isotopes: int, optional
    :param isotope_pmin: Minimal probability of an isotope to be considered, defaults to 0.01
    :type isotope_pmin: float, optional
    :param peak_width: Standard peak width, defaults to 60.0
    :type peak_width: float, optional
    :return: Feature map
    :rtype: oms.FeatureMap
    """
    if isinstance(experiment, str):
        name = experiment
    else:
        name = ""
    experiment = load_experiment(experiment)

    # FeatureMap to store results
    feature_map = oms.FeatureMap()
    
    # create FeatureFinderAlgorithmMetaboIdent and assign ms data
    ff = oms.FeatureFinderAlgorithmMetaboIdent()
    ff.setMSData(experiment)
    ff_par = ff.getDefaults()
    ff_par.setValue(b"extract:mz_window",  mz_window)
    if rt_window:
        ff_par.setValue(b"extract:rt_window",  rt_window)
    ff_par.setValue(b'extract:n_isotopes', n_isotopes)
    ff_par.setValue(b'extract:isotope_pmin', isotope_pmin)
    ff_par.setValue(b"detect:peak_width",  peak_width)
    ff.setParameters(ff_par)
    
    # run the FeatureFinderMetaboIdent with the metabo_table and mzML file path -> store results in fm
    ff.run(metab_table, feature_map, name)
    
    feature_map.setUniqueIds()  # Assigns a new, valid unique id per feature

    if feature_filepath:
        oms.FeatureXMLFile().store(feature_filepath, feature_map)
    
    return feature_map

def targeted_feature_detection(experiment: Union[oms.MSExperiment, str], compound_library_file:str,
                               mz_window:float=5.0, rt_window:Optional[float]=None, n_isotopes:int=2, isotope_pmin:float=0.01,
                               peak_width:float=60.0, mass_range:list=[50.0, 10000.0]) -> oms.FeatureMap:
    """
    Feature detection with a given metabolic table with compund library file.

    :param experiment: Input experiment
    :type experiment: Union[oms.MSExperiment, str]
    :param compound_library_file: Path to compound library file to define metabolic table from.
    :type compound_library_file: str
    :param metab_table: Metabilites table
    :type metab_table: list
    :param feature_filepath: Filpath to featureXML, defaults to None
    :type feature_filepath: Optional[str], optional
    :param mz_window: m/z window width, defaults to 5.0
    :type mz_window: float, optional
    :param rt_window: Retention time window width, defaults to None
    :type rt_window: Optional[float], optional
    :param n_isotopes: Number of considered isotopes, defaults to 2
    :type n_isotopes: int, optional
    :param isotope_pmin: Minimal probability of an isotope to be considered, defaults to 0.01
    :type isotope_pmin: float, optional
    :param peak_width: Standard peak width, defaults to 60.0
    :type peak_width: float, optional
    :return: Feature map
    :rtype: pyopenms.FeatureMap
    """
    experiment = load_experiment(experiment)
    
    print("Defining metabolite table...")
    metab_table = define_metabolite_table(compound_library_file, mass_range)
    print("Metabolite table defined...")
    
    feature_map = feature_detection_targeted(experiment=experiment, metab_table=metab_table, 
                                             mz_window=mz_window, rt_window=rt_window, peak_width=peak_width,
                                             n_isotopes=n_isotopes, isotope_pmin=isotope_pmin)
    print("Feature map created.")
    
    return feature_map


def targeted_features_detection(in_dir: str, run_dir:str, file_ending:str, compound_library_file:str, 
                                mz_window:float=5.0, rt_window:float=20.0, n_isotopes:int=2, isotope_pmin:float=0.01,
                                peak_width:float=60.0, mass_range:list=[50.0, 10000.0]) -> list[oms.FeatureMap]:
    """
    Feature detection with a given metabolic table with compund library file.

    :param in_dir: Input directory
    :type in_dir: str
    :param run_dir: Run directory
    :type run_dir: str
    :param file_ending: File ending
    :type file_ending: str
    :param compound_library_file: Path to compound library file to define metabolic table from.
    :type compound_library_file: str
    :param metab_table: Metabilites table
    :type metab_table: list
    :param feature_filepath: Filpath to featureXML, defaults to None
    :type feature_filepath: Optional[str], optional
    :param mz_window: m/z window width, defaults to 5.0
    :type mz_window: float, optional
    :param rt_window: Retention time window width, defaults to None
    :type rt_window: Optional[float], optional
    :param n_isotopes: Number of considered isotopes, defaults to 2
    :type n_isotopes: int, optional
    :param isotope_pmin: Minimal probability of an isotope to be considered, defaults to 0.01
    :type isotope_pmin: float, optional
    :param peak_width: Standard peak width, defaults to 60.0
    :type peak_width: float, optional
    :return: List of feature maps
    :rtype: list[pyopenms.FeatureMap]
    """
    print("Defining metabolite table...")
    metab_table = define_metabolite_table(compound_library_file, mass_range)
    print("Metabolite table defined...")

    feature_folder = clean_dir(run_dir, "features_targeted")
    feature_maps = []
    for file in tqdm(os.listdir(in_dir)):
        if file.endswith(file_ending):
            experiment_file = os.path.join(in_dir, file)
            feature_filepath = os.path.join(feature_folder, f"{file[:-len(file_ending)]}.featureXML")
            feature_map = feature_detection_targeted(experiment=experiment_file, metab_table=metab_table, feature_filepath=feature_filepath,
                                                     mz_window=mz_window, rt_window=rt_window, n_isotopes=n_isotopes,
                                                     isotope_pmin=isotope_pmin, peak_width=peak_width)
            feature_maps.append(feature_map)
          
    return feature_maps



## Clustering
def extract_from_clustering(df:pd.DataFrame, clustering) -> np.ndarray:
    """
    Extract mzs and intensities from clustering

    :param df: Input dataframe
    :type df: pandas.DataFrame
    :param clustering: Clustering
    :type clustering: np.ndarray
    :return: m/z + inty as paired array
    :rtype: np.ndarray
    """
    mzs = []
    intys = []
    for c in np.unique(clustering):
        indices = clustering == c
        intys_clust = df["inty"][indices].values
        mzs_clust = df["mz"][indices].values
        inty = np.max(intys_clust)
        mz = np.sum(mzs_clust * intys_clust**3) / np.sum(intys_clust**3)
        mzs.append(mz)
        intys.append(inty)
    return np.array( [mzs, intys] )    


def cluster_matlab(df:pd.DataFrame, height_lim:int=1000, prominence_lim:int=1000, threshold:float=(7e-2)**2):
    """
    Clusters according to FIA matlab routine

    :param df: Input dataframe
    :type df: pd.DataFrame
    :param height_lim: height limit, defaults to 1000
    :type height_lim: int, optional
    :param prominence_lim: prominence limit, defaults to 1000
    :type prominence_lim: int, optional
    :param threshold: threshold to cut off values, defaults to (7e-2)**2
    :type threshold: float, optional
    :return: m/z + inty as paired array
    :rtype: np.ndarray
    """
    # Peak detection
    peaks, *_ = find_peaks(df["inty"], height=height_lim, prominence=prominence_lim)    # type: ignore
    peaked_df = df.loc[df.index[peaks]]

    # Distance calculation @(x,y) (x(:,1)-y(:,1)).^2  + (x(:,2)==y(:,2))*10^6; 
    distances = pdist(peaked_df["mz"].values.reshape(-1,1), metric="cityblock")**2 + pdist(peaked_df["RT"].values.reshape(-1,1), metric="hamming")*1e6  # type: ignore  
    tree = linkage(distances, method="complete")
    clustering =  fcluster(tree, t=threshold, criterion="distance")
    return extract_from_clustering(peaked_df, clustering)


def cluster_sliding_window(comb_experiment:oms.MSExperiment, height_lim:int=1000, prominence_lim:int=1000, window_len:int=2000,
                           window_shift=1000, threshold:float=(7e-2)**2):
    """
    Applies clustering over sliding window in an experiment. The result may contain duplicates or close to duplicates.

    :param comb_experiment: Input experiment
    :type comb_experiment: pyopenms.MSExperiment
    param height_lim: height limit, defaults to 1000
    :type height_lim: int, optional
    :param prominence_lim: prominence limit, defaults to 1000
    :type prominence_lim: int, optional
    :param window_len: Window length, defaults to 2000
    :type window_len: int, optional
    :param window_shift: Window shift, defaults to 1000
    :type window_shift: int, optional
    :param threshold: threshold to cut off values, defaults to (7e-2)**2
    :type threshold: float, optional
    :return: Clustered experiment
    :rtype: pyopenms.MSExperiment
    """    
    n = len(comb_experiment.getSpectra()[0].get_peaks()[0])
    df = comb_experiment.get_df(long=True)
    peaks = []
    for i in tqdm(range(0, n - window_len, window_shift)):
        clust_exp = cluster_matlab(df.loc[i:i + window_len], height_lim=height_lim, prominence_lim=prominence_lim, threshold=threshold)
        peaks.append(clust_exp)
    clust_exp = cluster_matlab(df.loc[n-window_len:n], height_lim=height_lim, prominence_lim=prominence_lim, threshold=threshold)
    peaks = np.column_stack(peaks)
    peaks = np.unique(peaks, axis=1)

    # Packaging
    clustered_experiment = oms.MSExperiment()
    clustered_spectrum = oms.MSSpectrum()
    clustered_spectrum.set_peaks((peaks[0], peaks[1]))         # type: ignore
    clustered_experiment.addSpectrum(clustered_spectrum)
    return clustered_experiment



## Label assigning
### Accurate Mass
def accurate_mass_search(consensus_map:oms.ConsensusMap, database_dir:str, tmp_dir:str,
                         positive_adducts_file:str, negative_adducts_file:str, 
                         HMDBMapping_file:str, HMDB2StructMapping_file:str,
                         ionization_mode:str="auto") -> pd.DataFrame:
    """
    Assigning metbolites to consensus map masses.

    :param consensus_map: Input consensus map
    :type consensus_map: pyopenms.ConsensusMap
    :param database_dir: Database directory
    :type database_dir: str
    :param tmp_dir: Directory for temporary saves
    :type tmp_dir: str
    :param positive_adducts_file: File with possible positive adducts
    :type positive_adducts_file: str
    :param negative_adducts_file: File with possible negative adducts
    :type negative_adducts_file: str
    :param HMDBMapping_file: HMDBMapping file
    :type HMDBMapping_file: str
    :param HMDB2StructMapping_file: HMDB2StructMapping file
    :type HMDB2StructMapping_file: str
    :param ionization_mode: Ionization mode, defaults to "auto"
    :type ionization_mode: str, optional
    :return: Annotated dataframe
    :rtype: pandas.DataFrame
    """
    tmp_dir = clean_dir(tmp_dir)

    ams = oms.AccurateMassSearchEngine()

    ams_params = ams.getParameters()
    ams_params.setValue( "ionization_mode", ionization_mode)
    ams_params.setValue( "positive_adducts", os.path.join(database_dir, positive_adducts_file) )
    ams_params.setValue( "negative_adducts", os.path.join(database_dir, negative_adducts_file) )
    ams_params.setValue( "db:mapping", [os.path.join(database_dir, HMDBMapping_file)] )
    ams_params.setValue( "db:struct", [os.path.join(database_dir, HMDB2StructMapping_file)] )
    ams.setParameters(ams_params)

    mztab = oms.MzTab()
    ams.init()

    ams.run(consensus_map, mztab)

    oms.MzTabFile().store(os.path.join(tmp_dir, "ids.tsv"), mztab)

    with open( os.path.join(tmp_dir, "ids_smsection.tsv"), "w" ) as outfile, open( os.path.join(tmp_dir, "ids.tsv"), "r" ) as infile:
        for line in infile:
            if line.lstrip().startswith("SM"):
                outfile.write(line[4:])

    ams_df = pd.read_csv(os.path.join(tmp_dir, "ids_smsection.tsv"), sep="\t")

    return ams_df



# Transformation to DataFrame
def consensus_map_to_df(consensus_map:oms.ConsensusMap) -> pd.DataFrame:
    """
    Transforms a consensus map into a daraframe

    :param consensus_map: Input consensus map
    :type consensus_map: pyopenms.ConsensusMap
    :return: Dataframe from consensus map
    :rtype: pandas.DataFrame
    """    
    intensities = consensus_map.get_intensity_df()
    meta_data = consensus_map.get_metadata_df()[["RT", "mz", "quality"]]

    cm_df = pd.concat([meta_data, intensities], axis=1)
    cm_df.reset_index(drop=True, inplace=True)
    return cm_df 


# Consensus map filtering
def filter_consensus_map_df(consensus_map_df:pd.DataFrame, max_missing_values:int=1,
                            min_feature_quality:Optional[float]=0.8) -> pd.DataFrame:
    """
    Filter consensus map DataFrame according to missing values and feature quality.

    :param consensus_map_df: Input consensus map dataframe
    :type consensus_map_df: pandas.DataFrame
    :param max_missing_values: Maximum number of missing values, defaults to 1
    :type max_missing_values: int, optional
    :param min_feature_quality: Minimal quality of feature, defaults to 0.8
    :type min_feature_quality: Optional[float], optional
    :return: Filtered consensus map dataframe
    :rtype: pandas.DataFrame
    """
    to_drop = []
    cm_df = deepcopy(consensus_map_df)
    for i, row in cm_df.iterrows():
        if row.isna().sum() > max_missing_values:
            if min_feature_quality and row["quality"] < min_feature_quality:
                    to_drop.append(i)

    cm_df.drop(index=cm_df.index[to_drop], inplace=True)
    return cm_df


# Consensus map imputation
def impute_consensus_map_df(consensus_map_df:pd.DataFrame, n_nearest_neighbours:int=2) -> pd.DataFrame:
    """
    Data imputation with k-neares-neighbours (kNN).

    :param consensus_map_df: Input consensus map dataframe
    :type consensus_map_df: pandas.DataFrame
    :param n_nearest_neighbours: K nearest neighbours, defaults to 2
    :type n_nearest_neighbours: int, optional
    :return: Consensus map dataframe with imputed values
    :rtype: pandas.DataFrame
    """
    if len(consensus_map_df.index) > 0:
        imputer = Pipeline(
            [
                ( "imputer", KNNImputer(n_neighbors=n_nearest_neighbours)),
                ( "pandarizer", FunctionTransformer( lambda x: pd.DataFrame(x, columns=consensus_map_df.columns) ) )
            ]
        )
        consensus_map_df = pd.DataFrame(imputer.fit_transform(consensus_map_df))
    return consensus_map_df


# Merging
def find_close(df1:pd.DataFrame, df1_col, df2:pd.DataFrame, df2_col, tolerance=0.001):
    """
    Find close values in two dataframes.

    :param df1: Input dataframe 1
    :type df1: pandas.DataFrame
    :param df1_col: Input dataframe 1 matched column
    :type df1_col: all
    :param df2: Input dataframe 2
    :type df2: pandas.DataFrame
    :param df2_col: Input dataframe 2 matched column
    :type df2_col: all
    :param tolerance: Absolute tolerance in difference, defaults to 0.001
    :type tolerance: float, optional
    :yield: Datframe of close values
    :rtype: pandas.DataFrame
    """    
    for index, value in df1[df1_col].items():
        indices = df2.index[np.isclose(df2[df2_col].values, value, atol=tolerance)]
        s = pd.DataFrame(data={'idx1': index, 'idx2': indices.values})
        yield s


def merge_by_mz(id_df_1:pd.DataFrame, id_df_2:pd.DataFrame, mz_tolerance:float=1e-04) -> pd.DataFrame:
    """
    Merge dataframes by mz column.

    :param id_df_1: Input dataframe 1
    :type id_df_1: pandas.DataFrame
    :param id_df_2: Input dataframe 2
    :type id_df_2: pandas.DataFrame
    :param mz_tolerance: Tolerance of m/z deviation, defaults to 1e-04
    :type mz_tolerance: float, optional
    :return: _description_
    :rtype: pandas.DataFrame
    """    
    id_df = id_df_1.copy()
    df_idx = pd.concat(find_close(id_df_1, "mz", id_df_2, "mz", tolerance=mz_tolerance), ignore_index=True)
    for i, row in df_idx.iterrows():
        id_df.at[row["idx1"], "centroided_intensity"] = (id_df_1.at[row["idx1"], "centroided_intensity"] + id_df_2.at[row["idx2"], "centroided_intensity"]) / 2
    not_merged = [i for i in id_df_2.index if i not in df_idx["idx2"]]
    return pd.concat([id_df, id_df_2.loc[not_merged]]).reset_index(drop=True)



# Printing
def print_params(p):
    """
    Print parameters of pyopenms class.

    :param p: Parameters
    :type p: dict-like
    """
    if p.size():
        for i in p.keys():
            print( "Param:", i, "Value:", p[i], "Description:", p.getDescription(i) )
    else:
        print("no data available")



### Plotting ###
def quick_plot(spectrum: oms.MSSpectrum, xlim: Optional[List[float]] = None, ylim: Optional[List[float]] = None,
               plottype: str = "line", log:List[str]=[]) -> Figure:
    """
    Shows a plot of a spectrum between the defined borders

    :param spectrum: Input spectrum
    :type spectrum: pyopenms.MSSpectrum
    :param xlim: x-axis limits, defaults to None
    :type xlim: Optional[List[float]], optional
    :param ylim: y-axis limits, defaults to None
    :type ylim: Optional[List[float]], optional
    :param plottype: Type of plot [line|scatter], defaults to "line"
    :type plottype: str, optional
    :param log: Axes to log-transform [x,y], defaults to []
    :type log: List[str], optional
    :return: Figure
    :rtype: Figure
    """
    fig, ax1 = plt.subplots()
    if plottype == "line":
        ax2 = sns.lineplot(x=spectrum.get_peaks()[0], y=spectrum.get_peaks()[1], ax=ax1) # type: ignore
    elif plottype == "scatter":
        ax2 = sns.scatterplot(x=spectrum.get_peaks()[0], y=spectrum.get_peaks()[1], sizes=(20, 20), ax=ax1)  # type: ignore
    else:
        ax2 = sns.scatterplot(x=spectrum.get_peaks()[0], y=spectrum.get_peaks()[1], sizes=(20, 20), ax=ax1)  # type: ignore

    if xlim:
        ax2.set_xlim(xlim[0], xlim[1])
    if ylim:
        ax2.set_ylim(ylim[0], ylim[1])
    if "y" in log:
        plt.yscale('log')
    if "x" in log:
        plt.xscale('log')
    return fig


def sns_plot(x, y, hue=None, size=None, xlim: Optional[List[float]] = None, ylim: Optional[List[float]] = None,
            plottype: str = "line", log:List[str]=[], sizes:Optional[Tuple[int, int]]=(20,20), palette:str="hls",
            figsize:Optional[Tuple[int,int]] = (18, 5)) -> None:
    """
    Shows a plot of a spectrum between the defined borders.

    :param x: x-values
    :type x: array-like
    :param y: x-values
    :type y: array-like
    :param hue: Column to group x and y values, defaults to None
    :type hue: all, optional
    :param size: Size of elements, defaults to None
    :type size: float, optional
    :param xlim: x-axis limits, defaults to None
    :type xlim: Optional[List[float]], optional
    :param ylim: y-axis limits, defaults to None
    :type ylim: Optional[List[float]], optional
    :param plottype: Type of plot [line|scatter], defaults to "line"
    :type plottype: str, optional
    :param log: Axes to log-transform [x,y], defaults to []
    :type log: List[str], optional
    :param sizes: Sizes of points in relation to y-value, defaults to (20,20)
    :type sizes: Optional[Tuple[int, int]], optional
    :param palette: Color palette, defaults to "hls"
    :type palette: str, optional
    :param figsize: Figure size, defaults to (18, 5)
    :type figsize: Optional[Tuple[int,int]], optional
    """
    plt.figure(figsize = figsize)
    if plottype == "line":
        sns.lineplot(x=y, y=y, hue=hue, size=size) # type: ignore
    elif plottype == "scatter":
        sns.scatterplot(x=x, y=y, hue=hue, size=size, sizes=sizes, palette=palette)  # type: ignore
    else:
        sns.scatterplot(x=x, y=y, hue=hue, size=size, sizes=sizes, palette=palette)  # type: ignore

    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    if "y" in log:
        plt.yscale('log')
    if "x" in log:
        plt.xscale('log')
    plt.show()



def dynamic_plot(experiment: oms.MSExperiment, mode: str = "lines", log:List[str]=["x"]) -> None:
    """
    Shows an interactive plot of all spectra in the experiment. May take a long time for large datasets.
    Recommended after centroiding, or data reduction.

    :param experiment: Input experiment
    :type experiment: pyopenms.MSExperiment
    :param mode: Mode of display ["lines" | "markers" | "lines+markers" | other pyplot.graph_objects options], defaults to "lines"
    :type mode: str, optional
    :param log: Axes to log-transform [x,y], defaults to []
    :type log: List[str], optional
    """
    fig = go.Figure()
    for spectrum in experiment:
        trace = go.Scatter(x=spectrum.get_peaks()[0],
                           y=spectrum.get_peaks()[1],
                           mode=mode,
                           name=spectrum.getRT())
        fig.add_trace(trace)
    if "y" in log:
        fig.update_yaxes(type="log")
    if "x" in log:
        fig.update_xaxes(type="log")
    fig.update_layout(title='Superplot MSExperiment')
    fig.show()


def plot_mass_traces(mass_traces, sel=[0,100], x:str="rt", y:str="mz", z:str="int", threed:bool=True):
    """
    Plot mass traces along 3d plot.

    :param mass_traces: List of mass traces
    :type mass_traces: list
    :param sel: Selection of convex hull points, defaults to [0,100]
    :type sel: list, optional
    :param x: x-axis column, defaults to "rt"
    :type x: str, optional
    :param y: y-axis column, defaults to "mz"
    :type y: str, optional
    :param z: z-axis column, defaults to "int"
    :type z: str, optional
    :param threed: 3D plot, defaults to True
    :type threed: bool, optional
    :return: Plot
    :rtype: plotly-express plot
    """    
    dfs = []
    for i in range(sel[0], sel[1]):
        peak = len(mass_traces[i].getConvexhull().getHullPoints()) / 2 + 1
        hp = mass_traces[i].getConvexhull().getHullPoints()[0:int(peak)]
        hp = np.insert(hp, 2, mass_traces[i].getSmoothedIntensities(), axis=1)
        hp = np.insert(hp, 3, i, axis=1)
        dfs.append(pd.DataFrame(hp, columns=["rt", "mz", "int", "id"]))
    ch_df = pd.concat(dfs)
    if threed:
        return px.line_3d(ch_df, x=x, y=y, z=z, color="id")
    else:
        return px.line(ch_df, x=x, y=y, color="id", hover_data=z)


def plot_feature_map_rt_alignment(ordered_feature_maps:list, legend:bool=False) -> None:
    """
    Plot feature map retention time alignment

    :param ordered_feature_maps: Listof feature maps
    :type ordered_feature_maps: list
    :param legend: Display legend, defaults to False
    :type legend: bool, optional
    """    
    fig = plt.figure(figsize=(10, 5))

    ax = fig.add_subplot(1, 2, 1)
    ax.set_title("consensus map before alignment")
    ax.set_ylabel("m/z")
    ax.set_xlabel("RT")

    # use alpha value to display feature intensity
    ax.scatter(
        [feature.getRT() for feature in ordered_feature_maps[0]],
        [feature.getMZ() for feature in ordered_feature_maps[0]],
        alpha=np.asarray([feature.getIntensity() for feature in ordered_feature_maps[0]])
        / max([feature.getIntensity() for feature in ordered_feature_maps[0]]),
    )

    for fm in ordered_feature_maps[1:]:
        ax.scatter(
            [feature.getMetaValue("original_RT") for feature in fm],
            [feature.getMZ() for feature in fm],
            alpha=np.asarray([feature.getIntensity() for feature in fm])
            / max([feature.getIntensity() for feature in fm]),
        )

    ax = fig.add_subplot(1, 2, 2)
    ax.set_title("consensus map after alignment")
    ax.set_xlabel("RT")

    for fm in ordered_feature_maps:
        ax.scatter(
            [feature.getRT() for feature in fm],
            [feature.getMZ() for feature in fm],
            alpha=np.asarray([feature.getIntensity() for feature in fm])
            / max([feature.getIntensity() for feature in fm]),
        )

    fig.tight_layout()
    if legend:
        fig.legend(
            [fm.getMetaValue("spectra_data")[0].decode() for fm in ordered_feature_maps],
            loc="lower center",
        )
        
    plt.show()

def extract_feature_coord(feature:oms.Feature, mzs:np.ndarray, retention_times:np.ndarray, intensities:np.ndarray,
                          labels:np.ndarray, sub_feat:Optional[oms.Feature]=None) -> list:
    """
    Extract feature coordinates for plots

    :param feature: Input feature
    :type feature: oms.Feature
    :param mzs: m/z values
    :type mzs: np.ndarray
    :param retention_times: Retention times
    :type retention_times: np.ndarray
    :param intensities: Intensity values
    :type intensities: np.ndarray
    :param labels: Labels
    :type labels: np.ndarray
    :param sub_feat: Sub-features, defaults to None
    :type sub_feat: Optional[oms.Feature], optional
    :return: List of mzs, retention times intensities and matching labels for plots
    :rtype: list
    """    
    if sub_feat:
        for i, hull_point in enumerate(sub_feat.getConvexHulls()[0].getHullPoints()):
            mzs = np.append(mzs, sub_feat.getMZ())
            retention_times = np.append(retention_times, hull_point[0])
            intensities = np.append(intensities, hull_point[1])
            labels = np.append(labels, feature.getMetaValue('label'))
    else:    
        mzs = np.append(mzs, feature.getMZ())
        retention_times = np.append(retention_times, feature.getRT())
        intensities = np.append(intensities, feature.getIntensity())
        labels = np.append(labels, feature.getMetaValue("label"))
        
    return [mzs, retention_times, intensities, labels]


def plot_features_3D(feature_map:oms.FeatureMap, plottype:str="scatter") -> pd.DataFrame:
    """
    Represents found features in 3D

    :param feature_map: Input feature map
    :type feature_map: oms.FeatureMap
    :param plottype: Type of plot [scatter|surface|line], defaults to "scatter"
    :type plottype: str, optional
    :raises ValueError: Use ['surface','scatter','line']
    :return: Dataframe with 3d features
    :rtype: pandas.DataFrame
    """
    mzs = np.empty([0])
    retention_times = np.empty([0])
    intensities = np.empty([0])
    labels = np.empty([0])

    for feature in feature_map:
        if feature.getSubordinates():
            for i, sub_feat in enumerate(feature.getSubordinates()):
                mzs, retention_times, intensities, labels = extract_feature_coord(feature, mzs, retention_times, intensities, labels, sub_feat)
        else:
            mzs, retention_times, intensities, labels = extract_feature_coord(feature, mzs, retention_times, intensities, labels)

    df = pd.DataFrame({"m/z": mzs, "rt": retention_times, "intensity": intensities, "labels": labels})
    
    if plottype == "surface":
        fig = go.Figure(data=[go.Surface(z=df)])
        fig.update_layout(title='3D plot of features', autosize=True,
                    width=500, height=500,
                    xaxis_title="m/z", yaxis_title="rt",
                    margin=dict(l=65, r=50, b=65, t=90),
                    scene = {
                            "aspectratio": {"x": 1, "y": 1, "z": 0.2}
                            })
    elif plottype == "line":
        fig = px.line_3d(data_frame=df, x="m/z", y="rt", z="intensity", color="labels")
    elif plottype == "scatter":
        fig = px.scatter_3d(data_frame=df, x="m/z", y="rt", z="intensity", color="labels", size_max=1)
    else:
        raise ValueError(f"{plottype} is not a valid type of plot. Use ['surface','scatter','line']")
 
    if plottype:
        fig.update_traces(showlegend=False)        
        fig.show()
    
    return df


def plot_id_df(id_df:pd.DataFrame, x:str="RT", y:str="mz") -> None:
    """
    Scatterplot dataframe with identified metabolites.

    :param id_df: Input dataframe
    :type id_df: pd.DataFrame
    :param x: x-axis, defaults to "RT"
    :type x: str, optional
    :param y: y-axis, defaults to "mz"
    :type y: str, optional
    """    
    fig = px.scatter(id_df, x="RT", y="mz", hover_name="identifications")
    fig.update_layout(title="Consensus features with identifications (hover)")
    fig.show()
