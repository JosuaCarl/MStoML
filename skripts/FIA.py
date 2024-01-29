import os
import csv
import shutil
import requests
from copy import deepcopy
from tqdm import tqdm
from numpy import *
import pandas as pd
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

def clean_dir(dir_path:str, subfolder:str=None) -> str: 
    if subfolder:
        dir_path = os.path.join(dir_path, subfolder)
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.mkdir(dir_path)
    return dir_path


### FILES ###
# Loading
def read_experiment(filepath: str) -> oms.MSExperiment:
    """
    Read in MzXML or MzML File as a pyopenms experiment
    @path: String
    return: pyopenms.MSExperiment
    """
    experiment = oms.MSExperiment()
    if filepath.endswith(".mzML") or filepath.endswith(".MzML"):
        file = oms.MzMLFile()
        file.load(filepath, experiment)
    elif filepath.endswith(".mzXML") or filepath.endswith(".MzXML"):
        file = oms.MzXMLFile()
        file.load(filepath, experiment)
    else: 
        raise ValueError(f'Invalid ending of {filepath}. Must be in [".MzXML", ".mzXML", ".MzML", ".mzML"]')
    return experiment


def load_experiment(filepath:str, experiment:oms.MSExperiment=None) -> oms.MSExperiment:
    """
    If no experiment is given, loads and returns it from either .mzML or .mzXML file.
    """
    if experiment:
        return experiment
    else:
        return read_experiment(filepath)
    

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
    for file in os.listdir(path_to_featureXMLs):
        fm = read_feature_map_XML(os.path.join(path_to_featureXMLs, file))
        feature_maps.append(fm)
    return feature_maps


def define_metabolite_table(path_to_library_file:str, mass_range:list) -> list:
    """
    Read tsv file and create list of FeatureFinderMetaboIdentCompound
    """
    metaboTable = []
    with open(path_to_library_file, "r") as tsv_file:
        tsv_reader = csv.reader(tsv_file, delimiter="\t")
        next(tsv_reader)  # skip header
        for row in tsv_reader:
            # extract mass range from metabolites
            if float(row[2]) > mass_range[0] and float(row[2]) < mass_range[1]:
                metaboTable.append(
                    oms.FeatureFinderMetaboIdentCompound(
                        row[0],  # name
                        row[1],  # sum formula
                        float(row[2]),  # mass
                        [int(float(charge)) for charge in row[3].split(",")],  # charges
                        [float(rt) for rt in row[4].split(",")],  # RTs
                        [
                            float(rt_range) for rt_range in row[5].split(",")
                        ],  # RT ranges
                        [
                            float(iso_distrib) for iso_distrib in row[6].split(",")
                        ],  # isotope distributions
                    )
                )
    return metaboTable


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
                                 ones(len(df.index)),
                                 zeros(len(df.index)),
                                 zeros(len(df.index)))),
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
def annotate_consensus_map_df(consensus_map_df:pd.DataFrame, mass_search_df:pd.DataFrame, result_path:str) -> pd.DataFrame:
    id_df = consensus_map_df

    id_df["identifications"] = pd.Series(["" for x in range(len(id_df.index))])

    for rt, mz, description in zip(
        mass_search_df["retention_time"],
        mass_search_df["exp_mass_to_charge"],
        mass_search_df["description"],
    ):
        indices = id_df.index[
            isclose(id_df["mz"], float(mz), atol=1e-05)
            & isclose(id_df["RT"], float(rt), atol=1e-05)
        ].tolist()
        for index in indices:
            if description != "null":
                id_df.loc[index, "identifications"] += str(description) + ";"
    id_df["identifications"] = [
        item[:-1] if ";" in item else "" for item in id_df["identifications"]
    ]
    id_df.to_csv(result_path, sep="\t", index=False)
    return id_df


# Storing
def store_experiment(filepath:str, experiment: oms.MSExperiment) -> None:
    """
    Store Experiment as MzXML file
    @experiment: pyopenms.MSExperiment
    @filepath: string with path to savefile
    return: None
    """
    if filepath.endswith(".mzXML"):
        oms.MzXMLFile().store(filepath, experiment)
    elif filepath.endswith(".mzML"):
        oms.MzMLFile().store(filepath, experiment)
    else:
        oms.MzMLFile().store(filepath, experiment)

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


### DATA TRANSFORMATION ###
# Limiting
def limit_spectrum(spectrum: oms.MSSpectrum, mz_lower_limit: int | float, mz_upper_limit: int | float,
                   sample_size: int) -> oms.MSSpectrum:
    """
    Limits the range of the Spectrum to <mz_lower_limit> and <mz_upper_limit>. 
    Uniformly samples <sample_size> number of peaks from the spectrum (without replacement).
    Returns: openms spectrum
    """
    mzs, intensities = spectrum.get_peaks()

    lim = [searchsorted(mzs, mz_lower_limit, side='right'), searchsorted(mzs, mz_upper_limit, side='left')]

    mzs = mzs[lim[0]:lim[1]]
    intensities = intensities[lim[0]:lim[1]]

    idxs = range(len(mzs))
    new_spectrum = oms.MSSpectrum()
    if len(mzs) > sample_size:
        idxs = random.choice(idxs, size=sample_size, replace=False)
    new_spectrum.set_peaks((mzs[idxs], intensities[idxs]))

    return new_spectrum


def limit_experiment(filepath:str, experiment: oms.MSExperiment = None, mz_lower_limit: int | float=0, mz_upper_limit: int | float=10000,
                     sample_size: int =10e6, deepcopy: bool = False) -> oms.MSExperiment:
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
    experiment = load_experiment(filepath, experiment)

    if deepcopy:
        lim_exp = copy_experiment(experiment)
    else:
        lim_exp = experiment
    lim_exp.setSpectra(
        [limit_spectrum(spectrum, mz_lower_limit, mz_upper_limit, sample_size) for spectrum in experiment.getSpectra()])
    return lim_exp


# Smoothing
def smooth_spectra(experiment: oms.MSExperiment, gaussian_width: float, deepcopy: bool = False) -> oms.MSExperiment:
    """
    Apply a Gaussian filter to all spectra in an experiment
    @experiment: pyopenms.MSExperiment
    @gaussian_width: float
    @deepcopy: make a deep copy of the Experiment, so it is an independent object
    return: oms.MSExperiment
    """
    experiment = load_experiment(filepath, experiment)

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
def centroid_experiment(filepath:str, experiment: oms.MSExperiment = None, deepcopy: bool = False) -> oms.MSExperiment:
    """
    Reduce dataset to centroids
    @experiment: pyopenms.MSExperiment
    @deepcopy: make a deep copy of the Experiment, so it is an independent object
    return: 
    """
    experiment = load_experiment(filepath, experiment)

    accu_exp = oms.MSExperiment()
    oms.PeakPickerHiRes().pickExperiment(experiment, accu_exp, True)
    if deepcopy:
        centroid_exp = copy_experiment(experiment)
    else:
        centroid_exp = experiment
    centroid_exp.setSpectra(accu_exp.getSpectra())

    return centroid_exp


def centroid_batch(in_dir:str, run_dir:str, file_ending:str=".mzML") -> str:
    """
    Centroids a batch of experiments, extracted from files in a given directory with a given file ending (i.e. .mzML or .mzXML).
    Returns the new directors as path/centroids.
    """
    cleaned_dir = os.path.normpath( clean_dir(run_dir, "centroids") )

    for file in os.listdir(in_dir):
        if file.endswith(file_ending):
            centroided_exp = centroid_experiment(os.path.join(in_dir, file), deepcopy=deepcopy)
            oms.MzMLFile().store(os.path.join(cleaned_dir, f"{file.split('.')[0]}.mzML"), centroided_exp)

    return cleaned_dir


# Merging
def merge_experiment(filepath:str, experiment: oms.MSExperiment = None, block_size: int = None, deepcopy: bool = False) -> oms.MSExperiment:
    """
    Merge several spectra into one spectrum (useful for MS1 spectra to amplify signals along near retention times)
    @experiment: pyopenms.MSExperiment
    @block_size: int
    @deepcopy: make a deep copy of the Experiment, so it is an independent object
    return: 
    """
    experiment = load_experiment(filepath, experiment)

    if block_size is None:
        block_size = experiment.getNrSpectra()

    if deepcopy:
        merge_exp = copy_experiment(experiment)
    else:
        merge_exp = experiment
    merge_exp.setSpectra(experiment.getSpectra())
    merger = oms.SpectraMerger()
    if block_size:
        param = merger.getParameters()
        param.setValue("block_method:rt_block_size", block_size)
        merger.setParameters(param)
    merger.mergeSpectraBlockWise(merge_exp)
    return merge_exp


def merge_batch(in_dir:str, run_dir:str, file_ending:str=".mzML", block_size: int = None, deepcopy: bool = False) -> str:
    """
    Merge several spectra into one spectrum (useful for MS1 spectra to amplify signals along near retention times)
    @experiment: pyopenms.MSExperiment
    @block_size: int
    @deepcopy: make a deep copy of the Experiment, so it is an independent object
    return: 
    """
    cleaned_dir = os.path.normpath( clean_dir(run_dir, "merged") )

    for file in os.listdir(in_dir):
        if file.endswith(file_ending):
            merged_exp = merge_experiment(os.path.join(in_dir, file), block_size=block_size, deepcopy=deepcopy)
            oms.MzMLFile().store(os.path.join(cleaned_dir, file), merged_exp)

    return cleaned_dir

# Normalization
def normalize_spectra(filepath:str, experiment: oms.MSExperiment = None, normalization_method: str = "to_one",
                      deepcopy: bool = False) -> oms.MSExperiment:
    """
    Normalizes spectra
    @experiment: pyopenms.MSExperiment
    @normalization_method: "to_TIC" | "to_one" 
    @deepcopy: make a deep copy of the Experiment, so it is an independent object
    """
    experiment = load_experiment(filepath, experiment)

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


def deisotope_experiment(filepath:str, experiment: oms.MSExperiment = None, fragment_tolerance: float = 0.1, fragment_unit_ppm: bool = False,
                         min_charge: int = 1, max_charge: int = 3,
                         keep_only_deisotoped: bool = True, min_isopeaks: int = 2, max_isopeaks: int = 10,
                         make_single_charged: bool = True, annotate_charge: bool = True,
                         annotate_iso_peak_count: bool = True, use_decreasing_model: bool = True,
                         start_intensity_check: bool = False, add_up_intensity: bool = False,
                         deepcopy: bool = False):

    experiment = load_experiment(filepath, experiment)

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
def mass_trace_detection(filepath: str, experiment: oms.MSExperiment = None,
                         mass_error_ppm: float = 10.0, noise_threshold_int: float = 3000.0) -> list:
    """
    Mass trace detection
    """
    experiment = load_experiment(filepath, experiment)
    
    mass_traces = ([])
    mtd = oms.MassTraceDetection()
    mtd_par = (mtd.getDefaults())
    mtd_par.setValue("mass_error_ppm", mass_error_ppm)
    mtd_par.setValue("noise_threshold_int", noise_threshold_int)
    mtd.setParameters(mtd_par)
    mtd.run(experiment, mass_traces, 0)

    return mass_traces


def elution_peak_detection(mass_traces: list, width_filtering: str = "fixed") -> list:
    """
    Elution peak detection
    """
    mass_traces_deconvol = []
    epd = oms.ElutionPeakDetection()
    epd_par = epd.getDefaults()
    # The fixed setting filters out mass traces outside the [min_fwhm: 1.0, max_fwhm: 60.0] interval
    epd_par.setValue("width_filtering", width_filtering)
    epd.setParameters(epd_par)
    epd.detectPeaks(mass_traces, mass_traces_deconvol)
    if epd.getParameters().getValue("width_filtering") == "auto":
        mass_traces_final = []
        epd.filterByPeakWidth(mass_traces_deconvol, mass_traces_final)
    else:
        mass_traces_final = mass_traces_deconvol

    return mass_traces_final


def feature_detection_untargeted(filepath: str, experiment: oms.MSExperiment = None,
                                 mass_traces_deconvol: list = None, isotope_filtering_model="none",
                                 charge_lower_bound:int=1, charge_upper_bound:int=3, 
                                 remove_single_traces: str = "true", mz_scoring_by_elements: str = "false",
                                 report_convex_hulls: str = "true") -> oms.FeatureMap:
    """
    Untargeted feature detection
    """

    experiment = load_experiment(filepath, experiment)

    feature_map = oms.FeatureMap()  # output features
    chrom_out = []  # output chromatograms
    ffm = oms.FeatureFindingMetabo()

    ffm_par = ffm.getDefaults()
    print(type(charge_lower_bound))
    ffm_par.setValue("charge_lower_bound", charge_lower_bound)
    ffm_par.setValue("charge_upper_bound", charge_upper_bound)
    ffm_par.setValue("isotope_filtering_model", isotope_filtering_model)
    ffm_par.setValue("remove_single_traces", remove_single_traces)  # remove mass traces without satellite isotopic traces
    ffm_par.setValue("mz_scoring_by_elements", mz_scoring_by_elements)
    ffm_par.setValue("report_convex_hulls", report_convex_hulls)
    ffm.setParameters(ffm_par)

    ffm.run(mass_traces_deconvol, feature_map, chrom_out)
    feature_map.setUniqueIds()  # Assigns a new, valid unique id per feature
    feature_map.setPrimaryMSRunPath([filepath.encode()])

    return feature_map


def assign_feature_maps_polarity(feature_maps:list, scan_polarity:str=None) -> list:
    """
    Assigns the polarity to a list of feature maps, depending on "pos"/"neg" in file name.
    """
    for fm in feature_maps:
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


def align_retention_times(feature_maps: list, max_num_peaks_considered:int=-1,max_mz_difference:float=10.0, mz_unit:str="ppm" ) -> list:
    """
    Use as reference for alignment, the file with the largest number of features
    Works well if you have a pooled QC for example.
    Returns the aligned map at the first position
    """
    ref_index = argmax([fm.size() for fm in feature_maps])
    feature_maps.insert(0, feature_maps.pop(ref_index))


    aligner = oms.MapAlignmentAlgorithmPoseClustering()

    trafos = {}

    # parameter optimization
    aligner_par = aligner.getDefaults()
    aligner_par.setValue( "max_num_peaks_considered", max_num_peaks_considered )  # -1 = infinite
    aligner_par.setValue( "pairfinder:distance_MZ:max_difference", max_mz_difference )
    aligner_par.setValue( "pairfinder:distance_MZ:unit", "ppm" )
    aligner.setParameters(aligner_par)
    aligner.setReference(feature_maps[0])

    for feature_map in feature_maps[1:]:
        trafo = oms.TransformationDescription()  # save the transformed data points
        aligner.align(feature_map, trafo)
        trafos[feature_map.getMetaValue("spectra_data")[0].decode()] = trafo
        transformer = oms.MapAlignmentTransformer()
        transformer.transformRetentionTimes(feature_map, trafo, True)

    return feature_maps


def detect_adducts(feature_maps: list, potential_adducts:list=None) -> list:
    feature_maps_adducts = []
    for feature_map in feature_maps:
        mfd = oms.MetaboliteFeatureDeconvolution()
        mdf_par = mfd.getDefaults()
        if potential_adducts:
            mdf_par.setValue("potential_adducts", potential_adducts)
        mfd.setParameters(mdf_par)
        feature_map_adduct = oms.FeatureMap()
        mfd.compute(feature_map, feature_map_adduct, oms.ConsensusMap(), oms.ConsensusMap())
        feature_maps_adducts.append(feature_map_adduct)

    return feature_maps_adducts


def store_feature_maps(feature_maps: list, out_dir:str, ending:str) -> None:
    # Store the feature maps as featureXML files!
    clean_dir(out_dir)
    for feature_map in feature_maps:
        oms.FeatureXMLFile().store(os.path.join(out_dir, feature_map.getMetaValue("spectra_data")[0].decode()[:-len(ending)] + ".featureXML"),
                                   feature_map)

def separate_feature_maps_pos_neg(feature_maps:list) -> list:
    """
    Separate the feature maps into positively and negatively charged feature maps.
    """
    positive_features = []
    negative_features = []
    for fm in feature_maps:
        if fm.getMetaValue("scan_polarity") == "positive":
            positive_features.append(fm)
        elif fm.getMetaValue("scan_polarity") == "negative":
            negative_features.append(fm)
    return [positive_features, negative_features]

def consensus_features_linking(feature_maps: list, feature_grouper:str="QT") -> oms.ConsensusMap:
    if feature_grouper == "KD":
        feature_grouper = oms.FeatureGroupingAlgorithmKD()
    elif feature_grouper == "QT":
        feature_grouper = oms.FeatureGroupingAlgorithmQT()
    else:
        raise ValueError(f"{feature_grouper} is not in list of implemented feature groupers. Choose from ['KD','QT'].")

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
def untargeted_feature_detection(filepath: str,
                                 experiment: oms.MSExperiment = None,
                                 feature_filepath: str = None,
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
    experiment = load_experiment(filepath, experiment)
    experiment.sortSpectra(True)

    # Mass trace detection
    mass_traces = mass_trace_detection(filepath, experiment, mass_error_ppm, noise_threshold_int)

    # Elution Peak Detection
    mass_traces_deconvol = elution_peak_detection(mass_traces, width_filtering)

    # Feature finding
    feature_map = feature_detection_untargeted(filepath=filepath,
                                               experiment=experiment,
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

    for file in os.listdir(in_dir):
        if file.endswith(file_ending):
            experiment_file = os.path.join(in_dir, file)
            feature_file = os.path.join(feature_folder, f"{file[:-len(file_ending)]}.featureXML")
            feature_map = untargeted_feature_detection(filepath=experiment_file, experiment=None,
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
def feature_detection_targeted(filepath: str, metab_table:list, experiment: oms.MSExperiment = None,
                               mz_window:float=5.0, rt_window:float=20.0, n_isotopes:int=2, isotope_pmin:float=0.0,
                               peak_width:float=60.0) -> oms.FeatureMap:
    """
    Feature detection with a given metabolic table
    """

    experiment = load_experiment(filepath, experiment)

    # FeatureMap to store results
    feature_map = oms.FeatureMap()
    
    # create FeatureFinderAlgorithmMetaboIdent and assign ms data
    ff = oms.FeatureFinderAlgorithmMetaboIdent()
    ff.setMSData(experiment)
    ff_par = ff.getDefaults()
    ff_par.setValue(b"extract:mz_window",  mz_window)
    ff_par.setValue(b"extract:rt_window",  rt_window)
    ff_par.setValue(b'extract:n_isotopes', n_isotopes)
    ff_par.setValue(b'extract:isotope_pmin', isotope_pmin)
    ff_par.setValue(b"detect:peak_width",  peak_width)
    ff.setParameters(ff_par)
    
    # run the FeatureFinderMetaboIdent with the metabo_table and mzML file path -> store results in fm
    ff.run(metab_table, feature_map, filepath)
    
    feature_map.setUniqueIds()  # Assigns a new, valid unique id per feature
    feature_map.setPrimaryMSRunPath([filepath.encode()])
    
    return feature_map

def targeted_feature_detection(filepath: str, experiment:oms.MSExperiment, compound_library_file:str, 
                               mz_window:float=5.0, rt_window:float=20.0, peak_width:float=60.0,
                               mass_cutoff:list=[50.0, 10000.0]) -> oms.FeatureMap:
    """
    @mz_window: ppm
    @rt_window: s
    @peak_width: s
    returns: pyopenms.FeatureMap
    """
    experiment = load_experiment(filepath, experiment)
    
    print("Defining metabolite table...")
    metab_table = define_metabolite_table(compound_library_file, mass_range)
    print("Metabolite table defined...")
    
    feature_map = feature_detection_targeted("", metab_table, experiment, mz_window, rt_window, peak_width)
    print("Feature map created.")
    
    return feature_map


def targeted_features_detection(in_dir: str, run_dir:str, file_ending:str, compound_library_file:str, 
                                mz_window:float=5.0, rt_window:float=20.0, peak_width:float=60.0,
                                mass_cutoff:list=[50.0, 10000.0]) -> oms.FeatureMap:
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
    for file in os.listdir(in_dir):
        if file.endswith(file_ending):
            experiment_file = os.path.join(in_dir, file)
            feature_file = os.path.join(feature_folder, f"{file[:-len(file_ending)]}.featureXML")
            feature_map = feature_detection_targeted(filepath=experiment_file,
                                                        metab_table=metab_table, 
                                                        experiment=None,
                                                        mz_window=mz_window,
                                                        rt_window=rt_window,
                                                        peak_width=peak_width)
            print("Feature map created.")
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
                            min_feature_quality:float=0.8) -> pd.DataFrame:
    """
    Filter consensus map DataFrame according to min
    """
    to_drop = []
    cm_df = deepcopy(consensus_map_df)

    for i, row in cm_df.iterrows():
        if row.isna().sum() > max_missing_values or row["quality"] < min_feature_quality:
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
        consensus_map_df = imputer.fit_transform(consensus_map_df)
    return consensus_map_df


### Plotting ###
def quick_plot(spectrum: oms.MSSpectrum, xlim: [int | float, int | float] = None, plottype: str = "line") -> None:
    """
    Shows a plot of a spectrum between the defined borders
    @spectrum: pyopenms.MSSpectrum
    @xlim: list of two positions
    @plottype: "line" | "scatter"
    returns: None, but displays plot
    """
    if plottype == "line":
        ax = sns.lineplot(x=spectrum.get_peaks()[0], y=spectrum.get_peaks()[1])
    elif plottype == "scatter":
        ax = sns.scatterplot(x=spectrum.get_peaks()[0], y=spectrum.get_peaks()[1])
    else:
        ax = sns.scatterplot(x=spectrum.get_peaks()[0], y=spectrum.get_peaks()[1])

    if xlim:
        ax.set_xlim(xlim[0], xlim[1])
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
        alpha=asarray([feature.getIntensity() for feature in ordered_feature_maps[0]])
        / max([feature.getIntensity() for feature in ordered_feature_maps[0]]),
    )

    for fm in ordered_feature_maps[1:]:
        ax.scatter(
            [feature.getMetaValue("original_RT") for feature in fm],
            [feature.getMZ() for feature in fm],
            alpha=asarray([feature.getIntensity() for feature in fm])
            / max([feature.getIntensity() for feature in fm]),
        )

    ax = fig.add_subplot(1, 2, 2)
    ax.set_title("consensus map after alignment")
    ax.set_xlabel("RT")

    for fm in ordered_feature_maps:
        ax.scatter(
            [feature.getRT() for feature in fm],
            [feature.getMZ() for feature in fm],
            alpha=asarray([feature.getIntensity() for feature in fm])
            / max([feature.getIntensity() for feature in fm]),
        )

    fig.tight_layout()
    if legend:
        fig.legend(
            [fm.getMetaValue("spectra_data")[0].decode() for fm in ordered_feature_maps],
            loc="lower center",
        )
        
    plt.show()

def extract_feature_coord(feature:oms.Feature, mzs:array, retention_times:array, intensities:array, labels:array, sub_feat:oms.Feature) -> list:
    if sub_feat:
        for i, hull_point in enumerate(sub_feat.getConvexHulls()[0].getHullPoints()):
            mzs = append(mzs, sub_feat.getMZ())
            retention_times = append(retention_times, hull_point[0])
            intensities = append(intensities, hull_point[1])
            labels = append(labels, feature.getMetaValue('label'))
    else:    
        mzs = append(mzs, feature.getMZ())
        retention_times = append(retention_times, feature.getRT())
        intensities = append(intensities, feature.getIntensity())
        labels = append(labels, feature.getMetaValue("label"))
        

    return [mzs, retention_times, intensities, labels]

def plot_features_3D(feature_map:oms.FeatureMap, plottype:str=None) -> None:
    """
    Represents found features in 3D
    """
    mzs = empty([0])
    retention_times = empty([0])
    intensities = empty([0])
    labels = empty([0])

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
 
    if plottype:
        fig.update_traces(showlegend=False)        
        fig.show()
    
    return df

def plot_id_df(id_df:pd.DataFrame) -> None:
    fig = px.scatter(id_df, x="RT", y="mz", hover_name="identifications")
    fig.update_layout(title="Consensus features with identifications (hover)")
    fig.show()