import os
import stat
import time
from typing import overload, Any, List, Dict, Tuple, Set, Sequence, Union, Optional
import shutil
from click import Option
import requests
from copy import deepcopy
from tqdm import tqdm
import numpy as np
import pandas as pd
import scipy as sci
import pyopenms as oms
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from sklearn.impute import KNNImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline

### DATA OBTAIN ###
def batch_download(base_url:str, file_urls:list, save_directory:str) -> None:
    """
    Download files from a list into a directory.
    """
    for file_url in file_urls:
        request = requests.get(base_url + file_url, allow_redirects=True)
        open(os.path.join(save_directory, os.path.basename(file_url)), "wb").write(request.content)

### DIRECTORIES ###
def build_directory(dir_path:str) -> None:
    """
    Build a new directory in the given path.
    """
    if not os.path.isdir(os.path.join(os.getcwd(), dir_path)):
        os.mkdir(os.path.join(os.getcwd(), dir_path))

def clean_dir(dir_path:str, subfolder:Optional[str]=None) -> str: 
    if subfolder:
        dir_path = os.path.join(dir_path, subfolder)
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.mkdir(dir_path)
    return dir_path


### FILES ###
# Checking
def check_ending_experiment(file:str):
    return file.endswith(".mzML") or file.endswith(".MzML") or file.endswith(".mzXML") or file.endswith(".MzXML")

# Loading
def read_experiment(experiment_path: str) -> oms.MSExperiment:
    """
    Read in MzXML or MzML File as a pyopenms experiment
    @path: String
    return: pyopenms.MSExperiment
    """
    experiment = oms.MSExperiment()
    if experiment_path.endswith(".mzML") or experiment_path.endswith(".MzML"):
        file = oms.MzMLFile()
        file.load(experiment_path, experiment)
    elif experiment_path.endswith(".mzXML") or experiment_path.endswith(".MzXML"):
        file = oms.MzXMLFile()
        file.load(experiment_path, experiment)
    else: 
        raise ValueError(f'Invalid ending of {experiment_path}. Must be in [".MzXML", ".mzXML", ".MzML", ".mzML"]')
    return experiment


def load_experiment(experiment:Union[oms.MSExperiment, str]) -> oms.MSExperiment:
    """
    If no experiment is given, loads and returns it from either .mzML or .mzXML file.
    """
    if isinstance(experiment, oms.MSExperiment):
        return experiment
    else:
        return read_experiment(experiment)
    
def load_experiment_batch(experiments:Union[List[oms.MSExperiment|str], str], file_ending:Optional[str]=None) -> List[oms.MSExperiment]:
    """
    If no experiment is given, loads and returns it from either .mzML or .mzXML file.
    """
    if isinstance(experiments, str):
        if file_ending:
            return [load_experiment(os.path.join(experiments, file)) for file in tqdm(os.listdir(experiments)) if file.endswith(file_ending)]
        else:
            return [load_experiment(os.path.join(experiments, file)) for file in tqdm(os.listdir(experiments)) if check_ending_experiment(file)]
    else:
        return [load_experiment(experiment) for experiment in experiments]

def load_names_batch(directory:str, file_ending:str=".mzML") -> List[str]:
    """
    If no experiment is given, loads and returns it from either .mzML or .mzXML file.
    """
    return [file[0:-len(file_ending)] for file in tqdm(os.listdir(directory)) if file.endswith(file_ending)]
    
def load_fia_df(data_dir:str, file_ending:str):
    print("Loading experiments:")
    experiments = load_experiment_batch(data_dir)
    print("Loading names:")
    names = load_names_batch(data_dir, file_ending)
    samples = [name.split("_")[0] for name in names]
    polarities = [{"pos": 1, "neg": -1}.get(name.split("_")[-1]) for name in names]
    fia_df = pd.DataFrame([samples, polarities, experiments])
    fia_df = fia_df.transpose()
    fia_df.columns = ["sample", "polarity", "experiment"]
    return fia_df

def read_mnx(filepath: str) -> pd.DataFrame:
    """
    Read in chem_prop.tsv file from MetaNetX
    @filepath: path to file
    return: pandas.DataFrame
    """
    return pd.read_csv(filepath, sep="\t",
                       header=351, engine="pyarrow"
                       )[["#ID", "name", "formula", "charge", "mass"]].loc[1:].reset_index(drop=True).dropna()


def read_feature_map_XML(path_to_featureXML:str) -> oms.FeatureMap:
    """
    Reads in feature Map from file
    """
    fm = oms.FeatureMap()
    fh = oms.FeatureXMLFile()
    fh.load(path_to_featureXML, fm)
    return fm

def read_feature_maps_XML(path_to_featureXMLs:str) -> list:
    """
    Reads in feature Maps from file
    """
    feature_maps = []
    print("Reading in feature maps:")
    for file in tqdm(os.listdir(path_to_featureXMLs)):
        fm = read_feature_map_XML(os.path.join(path_to_featureXMLs, file))
        feature_maps.append(fm)
    return feature_maps


def define_metabolite_table(path_to_library_file:str, mass_range:list) -> list:
    """
    Read tsv file and create list of FeatureFinderMetaboIdentCompound
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


# Copying
def copy_experiment(experiment: oms.MSExperiment) -> oms.MSExperiment:
    """
    Makes a complete (recursive) copy of an experiment
    @experiment: pyopenms.MSExperiment
    return: pyopenms.MSExperiment
    """
    return deepcopy(experiment)


# Formatting
def mnx_to_oms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Turns a dataframe from MetaNetX into the required format by pyopenms for feature detection
    @df: pandas.DataFrame
    return: pandas.DataFrame
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
    Combines datframe with same <joiner>, while combining the name of <combiner> as the new index
    @df: pandas.DataFrame
    @joiner: string, that indicates the column that is the criterium for joining the rows
    @combiner: string, that indicates the column that should be combined as an identifier
    return: 
    """
    comb = pd.DataFrame(columns=df.columns)
    for j in tqdm(df[joiner].unique()):
        query = df.loc[df[joiner] == j]
        line = query.iloc[0].copy()
        line[combiner] = query[combiner].to_list()
        comb.loc[len(comb.index)] = line
    comb = comb.set_index(combiner)
    return comb

# Annotation
def annotate_consensus_map_df(consensus_map_df:pd.DataFrame, mass_search_df:pd.DataFrame, result_path:str=".",
                              mz_tolerance:float=1e-05) -> pd.DataFrame:
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
    Store Experiment as MzXML file
    @experiment: pyopenms.MSExperiment
    @experiment_path: string with path to savefile
    return: None
    """
    if experiment_path.endswith(".mzXML"):
        oms.MzXMLFile().store(experiment_path, experiment)
    elif experiment_path.endswith(".mzML"):
        oms.MzMLFile().store(experiment_path, experiment)
    else:
        oms.MzMLFile().store(experiment_path, experiment)

def store_feature_maps(feature_maps: list, out_dir:str, names: Union[list[str], str]=[], file_ending:str=".mzML") -> None:
    # Store the feature maps as featureXML files!
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


# Printing

def print_params(p):
    """
    Print all parameters
    """
    if p.size():
        for i in p.keys():
            print( "Param:", i, "Value:", p[i], "Description:", p.getDescription(i) )
    else:
        print("no data available")

### Compound tables transformation ###
def merge_compounds(path_to_tsv:str) -> pd.DataFrame:
	"""
	Joins entries with equal Mass and SumFormula.
	Links CompoundName with: ;
	Links rest with: ,
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


### MS DATA TRANSFORMATION ###
# Limiting
def limit_spectrum(spectrum: oms.MSSpectrum, mz_lower_limit: int | float, mz_upper_limit: int | float,
                   sample_size: int, statistic:str="sum") -> oms.MSSpectrum:
    """
    Limits the range of the Spectrum to <mz_lower_limit> and <mz_upper_limit>. 
    Uniformly samples <sample_size> number of peaks from the spectrum (without replacement).
    Returns: openms spectrum
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


def limit_experiment(experiment: Union[oms.MSExperiment, str], mz_lower_limit: int | float=0, mz_upper_limit: int | float=10000,
                     sample_size:int=100000, statistic:str="sum", deepcopy: bool = False) -> oms.MSExperiment:
    """
    Limits the range of all spectra in an experiment to <mz_lower_limit> and <mz_upper_limit>. 
    Uniformly samples <sample_size> number of peaks from the spectrum (without replacement).
    @experiment: pyopenms.MSExperiment
    @mz_lower_limit: number
    @mz_upper_limit: number
    @sample_size: int
    @deepcopy: make a deep copy of the Experiment, so it is an independent object
    Returns: pyopenms.MSExperiment 
    """
    experiment = load_experiment(experiment)

    if deepcopy:
        lim_exp = copy_experiment(experiment)
    else:
        lim_exp = experiment
    lim_exp.setSpectra(
        [limit_spectrum(spectrum, mz_lower_limit, mz_upper_limit, sample_size, statistic) for spectrum in experiment.getSpectra()])
    return lim_exp


# Combination
def combine_spectra(experiment:oms.MSExperiment) -> oms.MSSpectrum:
    spectrum = experiment.getSpectra()[0]
    intensities_all = np.zeros(spectrum.size())
    mzs_all = spectrum.get_peaks()[0]
    for spectrum in experiment.getSpectra():
        mzs, intensities = spectrum.get_peaks()
        intensities_all = np.sum([intensities_all, intensities], axis=0) # type: ignore
    comb_spectrum = oms.MSSpectrum()
    comb_spectrum.set_peaks( (mzs_all, intensities_all) ) # type: ignore
    return comb_spectrum


# Smoothing
def smooth_spectra(experiment: Union[oms.MSExperiment, str], gaussian_width: float, deepcopy: bool = False) -> oms.MSExperiment:
    """
    Apply a Gaussian filter to all spectra in an experiment
    @experiment: pyopenms.MSExperiment
    @gaussian_width: float
    @deepcopy: make a deep copy of the Experiment, so it is an independent object
    return: oms.MSExperiment
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
# Centroiding
def centroid_experiment(experiment: Union[oms.MSExperiment, str], instrument:str="Orbitrap",
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
    Reduce dataset to centroids
    @experiment: pyopenms.MSExperiment
    @deepcopy: make a deep copy of the Experiment, so it is an independent object
    return: 
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


def centroid_batch(in_dir:str, run_dir:str, file_ending:str=".mzML",
                   instrument:str="Orbitrap",
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
    """
    cleaned_dir = os.path.normpath( clean_dir(run_dir, "centroids") )

    for file in tqdm(os.listdir(in_dir)):
        if file.endswith(file_ending):
            centroided_exp = centroid_experiment(os.path.join(in_dir, file),
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
            oms.MzMLFile().store(os.path.join(cleaned_dir, f"{file.split('.')[0]}.mzML"), centroided_exp)

    return cleaned_dir


# Merging
def merge_experiment(experiment: Union[oms.MSExperiment, str], block_size: Optional[int] = None,
                     mz_binning_width:float=1.0, mz_binning_width_unit:str="ppm", average_gaussian_cutoff:float=0.01,
                     deepcopy: bool = False) -> oms.MSExperiment:
    """
    Merge several spectra into one spectrum (useful for MS1 spectra to amplify signals)
    @experiment: pyopenms.MSExperiment
    @block_size: int
    @deepcopy: make a deep copy of the Experiment, so it is an independent object
    return: 
    """
    experiment = load_experiment(experiment)

    if block_size is None:
        block_size = experiment.getNrSpectra()

    if deepcopy:
        merge_exp = copy_experiment(experiment)
    else:
        merge_exp = experiment
    merge_exp.setSpectra(experiment.getSpectra())
    merger = oms.SpectraMerger()
    param = merger.getParameters()
    param.setValue("mz_binning_width", mz_binning_width)
    param.setValue("mz_binning_width_unit", mz_binning_width_unit)
    param.setValue("average_gaussian:cutoff", average_gaussian_cutoff)
    param.setValue("block_method:rt_block_size", block_size)
    
    merger.setParameters(param)
    merger.mergeSpectraBlockWise(merge_exp)
    return merge_exp


def merge_batch(in_dir:str, run_dir:str, file_ending:str=".mzML", block_size: Optional[int] = None, 
                mz_binning_width:float=1.0, mz_binning_width_unit:str="ppm", average_gaussian_cutoff:float=0.01,
                deepcopy: bool = False) -> str:
    """
    Merge several spectra into one spectrum (useful for MS1 spectra to amplify signals along near retention times)
    @experiment: pyopenms.MSExperiment
    @block_size: int
    @deepcopy: make a deep copy of the Experiment, so it is an independent object
    return: 
    """
    cleaned_dir = os.path.normpath( clean_dir(run_dir, "merged") )

    for file in tqdm(os.listdir(in_dir)):
        if file.endswith(file_ending):
            merged_exp = merge_experiment(os.path.join(in_dir, file), block_size=block_size, 
                                          mz_binning_width=mz_binning_width, mz_binning_width_unit=mz_binning_width_unit,
                                          average_gaussian_cutoff=average_gaussian_cutoff,
                                          deepcopy=deepcopy)
            oms.MzMLFile().store(os.path.join(cleaned_dir, file), merged_exp)

    return cleaned_dir

# Normalization
def normalize_spectra(experiment: Union[oms.MSExperiment, str], normalization_method: str = "to_one",
                      deepcopy: bool = False) -> oms.MSExperiment:
    """
    Normalizes spectra
    @experiment: pyopenms.MSExperiment
    @normalization_method: "to_TIC" | "to_one" 
    @deepcopy: make a deep copy of the Experiment, so it is an independent object
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
                       start_intensity_check: bool = False, add_up_intensity: bool = False):
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


### Feature detection ###
def mass_trace_detection(experiment: Union[oms.MSExperiment, str],
                         mass_error_ppm: float = 10.0, noise_threshold_int: float = 1000.0, reestimate_mt_sd:str="true",
                         quant_method:str="median", trace_termination_criterion:str="outlier", trace_termination_outliers:int=3,
                         min_trace_length:float=5.0, max_trace_length:float=-1.0) -> list:
    """
    Mass trace detection
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

def mass_trace_detection_batch(experiments: Union[List[oms.MSExperiment|str], str], file_ending:str=".mzML", 
                               mass_error_ppm: float = 10.0, noise_threshold_int: float = 1000.0, reestimate_mt_sd:str="true",
                               quant_method:str="median", trace_termination_criterion:str="outlier", trace_termination_outliers:int=3,
                               min_trace_length:float=5.0, max_trace_length:float=-1.0) -> list:
    """
    Mass trace detection
    """
    mass_traces_all = []
    if isinstance(experiments, str):
            experiments = [os.path.join(experiments, file) for file in os.listdir(experiments) if file.endswith(file_ending)]
    for experiment in tqdm(experiments):
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
    Elution peak detection
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
    Elution peak detection
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
    Untargeted feature detection
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

def feature_detection_untargeted_batch(experiments:Union[List[oms.MSExperiment|str], str], file_ending:str=".mzML",
                                       mass_traces_deconvol_all: list[list] = [], isotope_filtering_model="metabolites (2% RMS)",
                                       local_rt_range:float=3.0, local_mz_range:float=5.0, 
                                       charge_lower_bound:int=1, charge_upper_bound:int=3,
                                       chrom_fwhm:float=10.0, report_summed_ints:str="true",
                                       enable_RT_filtering:str="false", mz_scoring_13C:str="false",
                                       use_smoothed_intensities:str="false", report_convex_hulls: str = "true",
                                       report_chromatograms:str="false", remove_single_traces: str = "true",
                                       mz_scoring_by_elements: str = "false", elements:str="CHNOPS") -> list[oms.FeatureMap]:
    feature_maps = []
    if isinstance(experiments, str):
        experiments = [os.path.join(experiments, file) for file in os.listdir(experiments) if file.endswith(file_ending)]
    for i, experiment in enumerate(tqdm(experiments)):
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


def detect_adducts(feature_maps: list, potential_adducts:Sequence[str | bytes]=[], q_try:str="feature", mass_max_diff:float=10.0, unit:str="ppm", max_minority_bound:int=3,
                   verbose_level:int=0) -> list:
    """
    Assigning adducts to peaks
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
    Works well if you have a pooled QC for example.
    Returns the aligned map at the first position
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
    Untargeted detection of features.
    @experiment: pyopenms.MSExperiment
    @mass_error_ppm: float, error of the mass in parts per million
    @noise_threshold_int: threshold for noise in the intensity
    @width_filtering
    @deepcopy
    return
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
    Feature detection with a given metabolic table
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
    @mz_window: ppm
    @rt_window: s
    @peak_width: s
    returns: pyopenms.FeatureMap
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
    @mz_window: ppm
    @rt_window: s
    @peak_width: s
    returns: pyopenms.FeatureMap
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



### Label assigning
# Accurate Mass
def accurate_mass_search(consensus_map:oms.ConsensusMap, database_dir:str, tmp_dir:str,
                         positive_adducts_file:str, negative_adducts_file:str, 
                         HMDBMapping_file:str, HMDB2StructMapping_file:str,
                         ionization_mode:str="auto") -> pd.DataFrame:
    """
    Com
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

def annotate_feature_map(feature_map:oms.FeatureMap, metabolite_mass) -> oms.FeatureMap:
    return feature_map

# Transformation to DataFrame
def consensus_map_to_df(consensus_map:oms.ConsensusMap) -> pd.DataFrame:
    intensities = consensus_map.get_intensity_df()
    meta_data = consensus_map.get_metadata_df()[["RT", "mz", "quality"]]

    cm_df = pd.concat([meta_data, intensities], axis=1)
    cm_df.reset_index(drop=True, inplace=True)
    return cm_df 

# Consensus map filtering
def filter_consensus_map_df(consensus_map_df:pd.DataFrame, max_missing_values:int=1,
                            min_feature_quality:Optional[float]=0.8) -> pd.DataFrame:
    """
    Filter consensus map DataFrame according to min
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
    Data imputation with KNN
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
def find_close(df1, df1_col, df2, df2_col, tolerance=0.001):
    for index, value in df1[df1_col].items():
        indices = df2.index[np.isclose(df2[df2_col].values, value, atol=tolerance)]
        s = pd.DataFrame(data={'idx1': index, 'idx2': indices.values})
        yield s


def merge_by_mz(id_df_1:pd.DataFrame, id_df_2:pd.DataFrame, mz_tolerance=1e-04):
    id_df = id_df_1.copy()
    df_idx = pd.concat(find_close(id_df_1, "mz", id_df_2, "mz", tolerance=mz_tolerance), ignore_index=True)
    for i, row in df_idx.iterrows():
        id_df.at[row["idx1"], "centroided_intensity"] = (id_df_1.at[row["idx1"], "centroided_intensity"] + id_df_2.at[row["idx2"], "centroided_intensity"]) / 2
    not_merged = [i for i in id_df_2.index if i not in df_idx["idx2"]]
    return pd.concat([id_df, id_df_2.loc[not_merged]]).reset_index(drop=True)


### Plotting ###
def quick_plot(spectrum: oms.MSSpectrum, xlim: Optional[List[float]] = None, ylim: Optional[List[float]] = None, plottype: str = "line") -> None:
    """
    Shows a plot of a spectrum between the defined borders
    @spectrum: pyopenms.MSSpectrum
    @xlim: list of two positions
    @plottype: "line" | "scatter"
    returns: None, but displays plot
    """
    if plottype == "line":
        ax = sns.lineplot(x=spectrum.get_peaks()[0], y=spectrum.get_peaks()[1]) # type: ignore
    elif plottype == "scatter":
        ax = sns.scatterplot(x=spectrum.get_peaks()[0], y=spectrum.get_peaks()[1], sizes=(40, 40))  # type: ignore
    else:
        ax = sns.scatterplot(x=spectrum.get_peaks()[0], y=spectrum.get_peaks()[1], sizes=(40, 40))  # type: ignore

    if xlim:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim:
        ax.set_ylim(ylim[0], ylim[1])
    plt.show()


def dynamic_plot(experiment: oms.MSExperiment, mode: str = "lines") -> None:
    """
    Shows an interactive plot of all spectra in the experiment. May take a long time for large datasets.
    Recommended after centroiding, or data reduction.
    @experiment: pyopenms.MSExperiment
    @mode: "lines" | "markers" | "lines+markers" | other pyplot.graph_objects options
    returns: None, but displays plot
    """
    fig = go.Figure()
    for spectrum in experiment:
        trace = go.Scatter(x=spectrum.get_peaks()[0],
                           y=spectrum.get_peaks()[1],
                           mode=mode,
                           name=spectrum.getRT())
        fig.add_trace(trace)
    fig.update_layout(title='Superplot MSExperiment')
    fig.show()

def plot_mass_traces(mass_traces, sel=[0,100], x:str="rt", y:str="mz", z:str="int", threed:bool=True):
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
    fig = px.scatter(id_df, x="RT", y="mz", hover_name="identifications")
    fig.update_layout(title="Consensus features with identifications (hover)")
    fig.show()
