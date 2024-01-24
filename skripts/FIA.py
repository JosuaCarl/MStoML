import os
import pandas as pd
import pyopenms as oms
from numpy import *
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from copy import deepcopy
from tqdm import tqdm
import csv


### FILES ###
# Loading
def read_mzxml(filepath: str) -> oms.MSExperiment:
    """
    Read in MzXML File as a pyopenms experiment
    @path: String
    return: pyopenms.MSExperiment
    """
    experiment = oms.MSExperiment()
    file = oms.MzXMLFile()
    file.load(filename=filepath, exp=experiment)
    return experiment


def read_mnx(filepath: str) -> pd.DataFrame:
    """
    Read in chem_prop.tsv file from MetaNetX
    @filepath: path to file
    return: pandas.DataFrame
    """
    return pd.read_csv(filepath, sep="\t",
                       header=351, engine="pyarrow"
                       )[["#ID", "name", "formula", "charge", "mass"]].loc[1:].reset_index(drop=True).dropna()


def read_feature_map_XML(path_to_featureXML):
    fm = oms.FeatureMap()
    fh = oms.FeatureXMLFile()
    fh.load(path_to_featureXML, fm)
    return fm


def define_metabolite_table(path_to_library_file):
    """
    read tsv file and create list of FeatureFinderMetaboIdentCompound
    """
    metaboTable = []
    with open(path_to_library_file, "r") as tsv_file:
        tsv_reader = csv.reader(tsv_file, delimiter="\t")
        next(tsv_reader)  # skip header
        for row in tsv_reader:
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
                                 zeros(len(df.index)),
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


# Storing
def store_experiment(experiment: oms.MSExperiment, filepath: str) -> None:
    """
    Store Experiment as MzXML file
    @experiment: pyopenms.MSExperiment
    @filepath: string with path to savefile
    return: None
    """
    oms.MzXMLFile().store(filepath, experiment)


### DataHandling ###
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


def limit_experiment(experiment: oms.MSExperiment, mz_lower_limit: int | float, mz_upper_limit: int | float,
                     sample_size: int, deepcopy: bool = False) -> oms.MSExperiment:
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
def centroid_experiment(experiment: oms.MSExperiment, deepcopy: bool = False) -> oms.MSExperiment:
    """
    Reduce dataset to centroids
    @experiment: pyopenms.MSExperiment
    @deepcopy: make a deep copy of the Experiment, so it is an independent object
    return: 
    """
    accu_exp = oms.MSExperiment()
    oms.PeakPickerHiRes().pickExperiment(experiment, accu_exp, True)
    if deepcopy:
        centroid_exp = copy_experiment(experiment)
    else:
        centroid_exp = experiment
    centroid_exp.setSpectra(accu_exp.getSpectra())

    return centroid_exp


# Merging
def merge_spectra(experiment: oms.MSExperiment, block_size: int = None, deepcopy: bool = False) -> oms.MSExperiment:
    """
    Merge several spectra into one spectrum (useful for MS1 spectra to amplify signals along near retention times)
    @experiment: pyopenms.MSExperiment
    @block_size: int
    @deepcopy: make a deep copy of the Experiment, so it is an independent object
    return: 
    """
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


# Normalization
def normalize_spectra(experiment: oms.MSExperiment, normalization_method: str = "to_one",
                      deepcopy: bool = False) -> oms.MSExperiment:
    """
    Normalizes spectra
    @experiment: pyopenms.MSExperiment
    @normalization_method: "to_TIC" | "to_one" 
    @deepcopy: make a deep copy of the Experiment, so it is an independent object
    """
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


def deisotope_experiment(experiment: oms.MSExperiment, fragment_tolerance: float = 0.1, fragment_unit_ppm: bool = False,
                         min_charge: int = 1, max_charge: int = 3,
                         keep_only_deisotoped: bool = True, min_isopeaks: int = 2, max_isopeaks: int = 10,
                         make_single_charged: bool = True, annotate_charge: bool = True,
                         annotate_iso_peak_count: bool = True, use_decreasing_model: bool = True,
                         start_intensity_check: bool = False, add_up_intensity: bool = False,
                         deepcopy: bool = False):
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
# Untargeted
def untargeted_feature_detection(filepath: str,
                                 experiment: oms.MSExperiment = None,
                                 feature_filepath: str = None,
                                 mass_error_ppm: float = 5.0,
                                 noise_threshold_int: float = 3000.0,
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
    if not experiment:
        experiment = oms.MSExperiment()
        oms.MzMLFile().load(filepath, experiment)

    experiment.sortSpectra(True)

    # Mass trace detection
    mass_traces = mass_trace_detection(filepath, experiment, mass_error_ppm, noise_threshold_int)

    # Elution Peak Detection
    mass_traces_deconvol = elution_peak_detection(mass_traces, width_filtering)

    # Feature finding
    fm = feature_detection_untargeted(filepath, experiment, mass_traces_deconvol, isotope_filtering_model,
                                      remove_single_traces, mz_scoring_by_elements, report_convex_hulls)

    if feature_filepath:
        oms.FeatureXMLFile().store(feature_filepath, fm)

    return fm


def mass_trace_detection(filepath: str, experiment: oms.MSExperiment = None,
                         mass_error_ppm: float = 10.0, noise_threshold_int: float = 3000.0) -> list:
    """
    Mass trace detection
    """
    if not experiment:
        experiment = oms.MSExperiment()
        oms.MzMLFile().load(filepath, experiment)

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
                                 remove_single_traces: str = "true", mz_scoring_by_elements: str = "false",
                                 report_convex_hulls: str = "true") -> oms.FeatureMap:
    """
    Feature detection
    """

    if not experiment:
        experiment = oms.MSExperiment()
        oms.MzMLFile().load(filepath, experiment)

    # feature detection
    feature_map = oms.FeatureMap()  # output features
    chrom_out = []  # output chromatograms
    ffm = oms.FeatureFindingMetabo()
    ffm_par = ffm.getDefaults()
    ffm_par.setValue("isotope_filtering_model", isotope_filtering_model)
    ffm_par.setValue("remove_single_traces",
                     remove_single_traces)  # remove mass traces without satellite isotopic traces
    ffm_par.setValue("mz_scoring_by_elements", mz_scoring_by_elements)
    ffm_par.setValue("report_convex_hulls", report_convex_hulls)

    ffm.setParameters(ffm_par)
    ffm.run(mass_traces_deconvol, feature_map, chrom_out)
    feature_map.setUniqueIds()  # Assigns a new, valid unique id per feature
    feature_map.setPrimaryMSRunPath([filepath.encode()])

    return feature_map


def align_retention_time(feature_maps: list) -> list:
    """
    Use as reference for alignment, the file with the largest number of features
    Works well if you have a pooled QC for example.
    """
    ref_index = feature_maps.index(sorted(feature_maps, key=lambda x: x.size())[-1])

    aligner = oms.MapAlignmentAlgorithmPoseClustering()

    trafos = {}

    # parameter optimization
    aligner_par = aligner.getDefaults()
    aligner_par.setValue("max_num_peaks_considered", -1)  # infinite
    aligner_par.setValue(
        "pairfinder:distance_MZ:max_difference", 10.0
    )  # Never pair features with larger m/z distance
    aligner_par.setValue("pairfinder:distance_MZ:unit", "ppm")
    aligner.setParameters(aligner_par)
    aligner.setReference(feature_maps[ref_index])

    for feature_map in feature_maps[:ref_index] + feature_maps[ref_index + 1:]:
        trafo = oms.TransformationDescription()  # save the transformed data points
        aligner.align(feature_map, trafo)
        trafos[feature_map.getMetaValue("spectra_data")[0].decode()] = trafo
        transformer = oms.MapAlignmentTransformer()
        transformer.transformRetentionTimes(feature_map, trafo, True)

    return feature_maps


def detect_adducts(feature_maps: list, potential_adducts=None) -> list:
    if not potential_adducts:
        potential_adducts = [b"H:+:0.4", b"Na:+:0.2", b"NH4:+:0.2", b"H-1O-1:+:0.1", b"H-3O-2:+:0.1"]
    feature_maps_adducts = []
    for feature_map in feature_maps:
        mfd = oms.MetaboliteFeatureDeconvolution()
        mdf_par = mfd.getDefaults()
        mdf_par.setValue("potential_adducts", potential_adducts)
        mfd.setParameters(mdf_par)
        feature_map_adduct = oms.FeatureMap()
        mfd.compute(feature_map, feature_map_adduct, oms.ConsensusMap(), oms.ConsensusMap())
        feature_maps_adducts.append(feature_map_adduct)

    return feature_maps_adducts


def store_feature_maps(feature_maps: list):
    # Store the feature maps as featureXML files!
    for feature_map in feature_maps:
        oms.FeatureXMLFile().store(feature_map.getMetaValue("spectra_data")[0].decode()[:-4] + "featureXML",
                                   feature_map)


def consensus_features(feature_maps: list) -> oms.ConsensusMap:
    feature_grouper = oms.FeatureGroupingAlgorithmKD()

    consensus_map = oms.ConsensusMap()
    file_descriptions = consensus_map.getColumnHeaders()

    for i, feature_map in enumerate(feature_maps):
        file_description = file_descriptions.get(i, oms.ColumnHeader())
        file_description.filename = os.path.basename(
            feature_map.getMetaValue("spectra_data")[0].decode()
        )
        file_description.size = feature_map.size()
        file_descriptions[i] = file_description

    feature_grouper.group(feature_maps, consensus_map)
    consensus_map.setColumnHeaders(file_descriptions)
    consensus_map.setUniqueIds()

    return consensus_map


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
