import datetime
import json
import pandas as pd
import pyopenms as oms
from numpy import *
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from copy import deepcopy
from tqdm import tqdm


### FILES ###
# Loading
def read_mzxml(filepath:str) -> oms.MSExperiment:
    """
    Read in MzXML File as an pyopenms experiment
    @path: String
    return: pyopenms.MSExperiment
    """
    experiment = oms.MSExperiment()
    file = oms.MzXMLFile()
    file.load(filename=filepath, exp=experiment)
    return experiment

def read_mnx(filepath:str) -> pd.DataFrame:
    """
    Read in chem_prop.tsv file from MetaNetX
    @filepath: path to file
    return: pandas.DataFrame
    """
    return pd.read_csv(filepath, sep="\t",
                        header=351,  engine="pyarrow"
                        )[["#ID", "name", "formula", "charge", "mass"]].loc[1:].reset_index(drop=True).dropna()

def read_feature_map_XML(path_to_featureXML):
    fm = oms.FeatureMap()
    fh = oms.FeatureXMLFile()
    fh.load(path_to_featureXML, fm)
    return fm


# Copying
def copy_experiment(experiment:oms.MSExperiment) -> oms.MSExperiment:
    """
    Makes a complete (recursive) copy of an experiment
    @experiment: pyopenms.MSExperiment
    return: pyopenms.MSExperiment
    """
    return deepcopy(experiment)

# Formatting
def mnx_to_oms(df:pd.DataFrame) -> pd.DataFrame:
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
                            columns=["CompoundName", "SumFormula", "Mass", "Charge", "RetentionTime", "RetentionTimeRange", "IsotopeDistribution"])

def join_df_by(df:pd.DataFrame, joiner:str, combiner:str) -> pd.DataFrame:
    """
    Combines datframe with same <joiner>, while combining the name of <combiner> as the new index
    @df: pandas.DataFrame
    @joiner: string, that indicates the column that is the criterium for joining the rows
    @combiner: string, that indicates the column that should be combined as an identifier
    return: 
    """
    comb = pd.DataFrame(columns=df.columns)
    data_cols = list(df.columns).remove(joiner)
    for j in tqdm(df[joiner].unique()):
        query = df.loc[df[joiner] == j]
        line = query.iloc[0].copy()
        line[combiner] = list(query[combiner].values)
        comb.loc[len(comb.index)] = line
    comb = comb.set_index(combiner)
    return comb    

# Storing
def store_experiment(experiment:oms.MSExperiment, filepath:str) -> None:
    """
    Store Experiment as MzXML file
    @experiment: pyopenms.MSExperiment
    @filepath: string with path to savefile
    return: None
    """
    oms.MzXMLFile().store(filepath, experiment)


### DataHandling ###
# Limiting
def limit_spectrum(spectrum:oms.MSSpectrum, mz_lower_limit:int|float, mz_upper_limit:int|float, sample_size:int) -> oms.MSSpectrum:
    """
    Limits the range of the Spectrum to <mz_lower_limit> and <mz_upper_limit>. 
    Uniformly samples <sample_size> number of peaks from the spectrum (without replacement).
    Returns: openms spectrum
    """
    mzs, intensities = spectrum.get_peaks()

    lim = [searchsorted(mzs,mz_lower_limit, side='right'), searchsorted(mzs, mz_upper_limit, side='left')]

    mzs = mzs[lim[0]:lim[1]]
    intensities = intensities[lim[0]:lim[1]]

    idxs = range(len(mzs))
    new_spectrum = oms.MSSpectrum()
    if len(mzs) > sample_size:
        idxs = random.choice(idxs, size=sample_size, replace=False)
    new_spectrum.set_peaks( (mzs[idxs], intensities[idxs]) )

    return new_spectrum

def limit_experiment(experiment:oms.MSExperiment, mz_lower_limit:int|float, mz_upper_limit:int|float, sample_size:int, deepcopy:bool=False) -> oms.MSExperiment:
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
    lim_exp.setSpectra( [limit_spectrum(spectrum, mz_lower_limit, mz_upper_limit, sample_size) for spectrum in experiment.getSpectra()] )
    return lim_exp


# Smoothing
def smooth_spectra(experiment:oms.MSExperiment, gaussian_width:float, deepcopy:bool=False) -> oms.MSExperiment:
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
def centroid_experiment(experiment:oms.MSExperiment, deepcopy:bool=False) -> oms.MSExperiment:
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
def merge_spectra(experiment:oms.MSExperiment, block_size:int=None, deepcopy:bool=False) -> oms.MSExperiment:
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
def normalize_spectra(experiment:oms.MSExperiment, normalization_method:str="to_one", deepcopy:bool=False) -> oms.MSExperiment:
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
def deisotope_spectrum(spectrum:oms.MSSpectrum, fragment_tolerance:float=0.1, fragment_unit_ppm:bool=False, min_charge:int=1, max_charge:int=3,
                    keep_only_deisotoped:bool=True, min_isopeaks:int=2, max_isopeaks:int=10, make_single_charged:bool=True, annotate_charge:bool=True,
                    annotate_iso_peak_count:bool=True, use_decreasing_model:bool=True, start_intensity_check:bool=False, add_up_intensity:bool=False):
    spectrum.setFloatDataArrays( [] )

    oms.Deisotoper.deisotopeAndSingleCharge(
        spectra = spectrum, 
        fragment_tolerance = fragment_tolerance,
        fragment_unit_ppm = fragment_unit_ppm,
        min_charge = min_charge,
        max_charge = max_charge,
        keep_only_deisotoped = keep_only_deisotoped,
        min_isopeaks = min_isopeaks,
        max_isopeaks = max_isopeaks,
        make_single_charged = make_single_charged,
        annotate_charge = annotate_charge,
        annotate_iso_peak_count = annotate_iso_peak_count,
        use_decreasing_model = use_decreasing_model,
        start_intensity_check = start_intensity_check,
        add_up_intensity = add_up_intensity
    )

    return spectrum

def deisotope_experiment(experiment:oms.MSExperiment, fragment_tolerance:float=0.1, fragment_unit_ppm:bool=False, min_charge:int=1, max_charge:int=3,
                        keep_only_deisotoped:bool=True, min_isopeaks:int=2, max_isopeaks:int=10, make_single_charged:bool=True, annotate_charge:bool=True,
                        annotate_iso_peak_count:bool=True, use_decreasing_model:bool=True, start_intensity_check:bool=False, add_up_intensity:bool=False,
                        deepcopy:bool=False):
    if deepcopy:
        deisotop_exp = copy_experiment(experiment)
    else:
        deisotop_exp = experiment
    for i, spectrum in enumerate(deisotop_exp):
        deisotop_exp[i] = deisotope_spectrum(spectrum, fragment_tolerance, fragment_unit_ppm, min_charge, max_charge, keep_only_deisotoped,
                                            min_isopeaks, max_isopeaks, make_single_charged, annotate_charge, annotate_iso_peak_count,
                                            use_decreasing_model, start_intensity_check, add_up_intensity)
    return deisotop_exp


### Feature detection ###
# Untargeted
def untargeted_feature_detection(experiment:oms.MSExperiment,
                                 filepath:str,
                                 mass_error_ppm:float=5.0,
                                 noise_threshold_int:float=3000.0,
                                 width_filtering:str="fixed",
                                 isotope_filtering_model="none",
                                 remove_single_traces="true",
                                 mz_scoring_by_elements="false",
                                 report_convex_hulls="true",
                                 deepcopy:bool=False) -> oms.FeatureMap:
    """
    Untargeted detection of features.
    @experiment: pyopenms.MSExperiment
    @mass_error_ppm: float, error of the mass in parts per million
    @noise_threshold_int: threshold for noise in the intensity
    @width_filtering
    @deepcopy
    return
    """
    experiment.sortSpectra(True)

    # Mass trace detection
    mass_traces = []
    mtd = oms.MassTraceDetection()
    mtd_params = mtd.getDefaults()
    mtd_params.setValue("mass_error_ppm", mass_error_ppm)  # set according to your instrument mass error
    mtd_params.setValue("noise_threshold_int", noise_threshold_int)  # adjust to noise level in your data
    mtd.setParameters(mtd_params)
    mtd.run(experiment, mass_traces, 0)

    # Elution Peak Detection
    mass_traces_split = []
    epd = oms.ElutionPeakDetection()
    epd_params = epd.getDefaults()
    epd_params.setValue("width_filtering", width_filtering)
    epd.setParameters(epd_params)
    epd.detectPeaks(mass_traces, mass_traces_split)
    if epd.getParameters().getValue("width_filtering") == "auto":
        mass_traces_final = []
        epd.filterByPeakWidth(mass_traces_split, mass_traces_final)
    else:
        mass_traces_final = mass_traces_split

    # Feature finding
    fm = oms.FeatureMap()
    feat_chrom = []
    ffm = oms.FeatureFindingMetabo()
    ffm_params = ffm.getDefaults()
    ffm_params.setValue( "isotope_filtering_model", isotope_filtering_model )
    ffm_params.setValue( "remove_single_traces", remove_single_traces )  # set false to keep features with only one mass trace
    ffm_params.setValue( "mz_scoring_by_elements", mz_scoring_by_elements )
    ffm_params.setValue( "report_convex_hulls", report_convex_hulls )
    ffm.setParameters(ffm_params)
    ffm.run(mass_traces_final, fm, feat_chrom)
    fm.setUniqueIds()

    if filepath:
        oms.FeatureXMLFile().store(filepath, fm)

    return fm    
  
### Plotting ###
def quick_plot(spectrum:oms.MSSpectrum, xlim:[int|float, int|float]=None, plottype:str="line") -> None:
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
    if xlim:
        ax.set_xlim(xlim[0], xlim[1])
    plt.show()


def dynamic_plot(experiment:oms.MSExperiment, mode:str="lines") -> None:
    """
    Shows an interactive plot of all spectra in the experiment. May take a long time for large datasets. Recommendes after centroiding, or data reduction.
    @experiment: pyopenms.MSExperiment
    @mode: "lines" | "markers" | "lines+markers" | other pyplot.graph_objects options
    returns: None, but displays plot
    """
    fig = go.Figure()
    for spectrum in experiment:
        trace = go.Scatter( x = spectrum.get_peaks()[0],
                            y = spectrum.get_peaks()[1],
                            mode = 'lines',
                            name = spectrum.getRT())
        fig.add_trace(trace)
    fig.update_layout(title='Superplot MSExperiment')
    fig.show()

