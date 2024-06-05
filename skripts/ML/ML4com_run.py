#!/usr/bin/env python3

# imports
import sys
sys.path.append( '..' )

from FIA.FIA import *
from ML.ML4com import *
from VAE.vae import *

import warnings
warnings.simplefilter(action='ignore', category=sklearn.exceptions.UndefinedMetricWarning)

start_dir = "../.."
source = "annotated"

n_trials = 200
inner_fold = 3
outer_fold = 5
verbosity = 0



orig_dir = os.path.normpath(os.path.join(os.getcwd(), f"{start_dir}/data/Com8_grown_together"))
data_dir  = os.path.normpath(os.path.join(os.getcwd(), f"{start_dir}/runs/FIA/Com8_grown_together"))

strains = pd.read_csv( os.path.join(orig_dir, "strains.tsv"), sep="\t")
comm = pd.read_csv( os.path.join(orig_dir, "comb_one_hot.tsv"), sep="\t")
metData = pd.read_csv( os.path.join(orig_dir, "metData.tsv"), sep="\t")
metData.index = pd.read_csv( os.path.join(orig_dir, "metName.tsv"), sep="\t")

ys = comm
targets = strains["Organism"].values

if source == "latent":
    run_dir = os.path.normpath(os.path.join(os.getcwd(), f"{start_dir}/runs/ML/latent"))

    backend_name = "tensorflow"
    computation = "cpu"
    name = "cos"
    project = f"vae_{backend_name}_{computation}_{name}"
    outdir = Path( os.path.normpath( os.path.join(run_dir, project)) )
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    X = pd.read_csv(f"{start_dir}/runs/VAE/results/encoded_mu_{name}.tsv", index_col=0, sep="\t")
elif source == "annotated":
    run_dir = os.path.normpath(os.path.join(os.getcwd(), f"{start_dir}/runs/ML/annot"))
    name = "annotated_com8_grown_together"

    met_raw_pos = pd.read_excel( os.path.join( orig_dir, "FIA-Data Com8_20230717_P0024_msAV206-312.xlsx" ), sheet_name="pos" )
    met_raw_neg = pd.read_excel( os.path.join( orig_dir, "FIA-Data Com8_20230717_P0024_msAV206-312.xlsx" ), sheet_name="neg" )
    met_raw_comb = pd.concat( [ total_ion_count_normalization( join_df_metNames(met_raw_pos) ),
                                total_ion_count_normalization( join_df_metNames(met_raw_neg) ) ] )

    outdir = Path( os.path.normpath( os.path.join( run_dir, name) ) )
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    X = met_raw_comb.transpose()

from sklearn.neighbors import KNeighborsClassifier 
classifier = KNeighborsClassifier
algorithm_name = "Kneighbours classifier"

configuration_space = ConfigurationSpace()
hyperparameters = [ 
                    Constant("n_neighbors",   2),
                    Categorical("weights",    ["uniform", "distance"], default="uniform"),
                    Integer("leaf_size",      (10, 100), log=True, default=30),
                    Integer("p",              (1, 2), default=2),
                    Constant("metric",        "minkowski")
                  ]
configuration_space.add_hyperparameters( hyperparameters ) 

metrics_df, organism_metrics_df, overall_metrics_df = nested_cross_validate_model_sklearn( X=X, ys=ys, labels=targets, classifier=classifier,
                                                                                           configuration_space=configuration_space, n_trials=n_trials,
                                                                                           name=name, algorithm_name=algorithm_name, outdir=outdir,
                                                                                           fold=StratifiedKFold(n_splits=outer_fold), inner_fold=inner_fold, verbosity=verbosity)

plot_metrics_df(metrics_df, organism_metrics_df, overall_metrics_df, algorithm_name, outdir, show=False)



from sklearn.svm import SVC 
classifier = SVC
algorithm_name = "Support-vector classifier"

configuration_space = ConfigurationSpace()
hyperparameters = [ 
                    Float("C",              (0.5, 2.0), default=1.0),
                    Categorical("kernel",   ["linear", "poly", "rbf", "sigmoid"], default="rbf"),
                    Integer("degree",       (2,4), default=3),
                    Categorical("gamma",    ["scale", "auto"]),
                    Float("coef0",          (1e-12, 1.0), log=True, default=0.01),
                    Constant("random_state", 42),
                  ]
conditions = [
                InCondition(hyperparameters[2], hyperparameters[1], ["poly"]),
                InCondition(hyperparameters[4], hyperparameters[1], ["poly", "sigmoid"]),
                InCondition(hyperparameters[3], hyperparameters[1], ["rbf", "poly", "sigmoid"])
             ]

configuration_space.add_hyperparameters( hyperparameters ) 
configuration_space.add_conditions(conditions)

metrics_df, organism_metrics_df, overall_metrics_df = nested_cross_validate_model_sklearn( X=X, ys=ys, labels=targets, classifier=classifier,
                                                                                           configuration_space=configuration_space, n_trials=n_trials,
                                                                                           name=name, algorithm_name=algorithm_name, outdir=outdir,
                                                                                           fold=StratifiedKFold(n_splits=outer_fold), inner_fold=inner_fold, verbosity=verbosity)

plot_metrics_df(metrics_df, organism_metrics_df, overall_metrics_df, algorithm_name, outdir, show=False)



from sklearn.gaussian_process import GaussianProcessClassifier
classifier = GaussianProcessClassifier
algorithm_name = "Gaussian process classifier"


configuration_space = ConfigurationSpace()
hyperparameters = [ 
                    Constant("random_state", 42),
                    Integer("max_iter_predict",   (10, 1000), log=True, default=100),
                  ]
configuration_space.add_hyperparameters( hyperparameters )

metrics_df, organism_metrics_df, overall_metrics_df = nested_cross_validate_model_sklearn( X=X, ys=ys, labels=targets, classifier=classifier,
                                                                                           configuration_space=configuration_space, n_trials=n_trials,
                                                                                           name=name, algorithm_name=algorithm_name, outdir=outdir,
                                                                                           fold=StratifiedKFold(n_splits=outer_fold), inner_fold=inner_fold, verbosity=verbosity)

plot_metrics_df(metrics_df, organism_metrics_df, overall_metrics_df, algorithm_name, outdir, show=False)



from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB
algorithm_name = "Gaussian Naive-Bayes"

configuration_space = ConfigurationSpace()
hyperparameters = [ 
                    Constant("var_smoothing",   1e-9),
                  ]
configuration_space.add_hyperparameters( hyperparameters )

metrics_df, organism_metrics_df, overall_metrics_df = nested_cross_validate_model_sklearn( X=X, ys=ys, labels=targets, classifier=classifier,
                                                                                           configuration_space=configuration_space, n_trials=1,
                                                                                           name=name, algorithm_name=algorithm_name, outdir=outdir,
                                                                                           fold=StratifiedKFold(n_splits=outer_fold), inner_fold=inner_fold, verbosity=verbosity)

plot_metrics_df(metrics_df, organism_metrics_df, overall_metrics_df, algorithm_name, outdir, show=False)



from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier
algorithm_name = "Decision tree"

configuration_space = ConfigurationSpace()
ccp_alpha   = Float("ccp_alpha", (1e-3, 1e0), log=True, default=0.01)
configuration_space.add_hyperparameters([ccp_alpha])

metrics_df, organism_metrics_df, overall_metrics_df = nested_cross_validate_model_sklearn( X=X, ys=ys, labels=targets, classifier=classifier,
                                                                                           configuration_space=configuration_space, n_trials=n_trials,
                                                                                           name=name, algorithm_name=algorithm_name, outdir=outdir,
                                                                                           fold=StratifiedKFold(n_splits=outer_fold), inner_fold=inner_fold, verbosity=verbosity)

plot_metrics_df(metrics_df, organism_metrics_df, overall_metrics_df, algorithm_name, outdir, show=False)



from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier
algorithm_name = "Random forest"

configuration_space = ConfigurationSpace()
ccp_alpha       = Float("ccp_alpha", (1e-3, 1e-1), log=True, default=0.01)
n_estimators    = Integer("n_estimators", (10,1000), log=True, default=100)
max_depth       = Integer("max_depth", (5, 100), default=20)
configuration_space.add_hyperparameters([ccp_alpha, n_estimators, max_depth])

metrics_df, organism_metrics_df, overall_metrics_df = nested_cross_validate_model_sklearn( X=X, ys=ys, labels=targets, classifier=classifier,
                                                                                           configuration_space=configuration_space, n_trials=n_trials,
                                                                                           name=name, algorithm_name=algorithm_name, outdir=outdir,
                                                                                           fold=StratifiedKFold(n_splits=outer_fold), inner_fold=inner_fold, verbosity=verbosity)

plot_metrics_df(metrics_df, organism_metrics_df, overall_metrics_df, algorithm_name, outdir, show=False)



from xgboost import XGBClassifier
classifier = XGBClassifier
algorithm_name = "Extreme gradient boosting RF"

configuration_space = ConfigurationSpace()
objective           = Constant( "objective",            "binary:logistic")
num_parallel_tree   = Constant( "num_parallel_tree",    4)
n_estimators        = Integer(  "n_estimators",         (10,1000), log=True, default=100)
max_depth           = Integer(  "max_depth",            (1, 100), default=20)
subsample           = Float(    "subsample",            (1e-1, 1e0), default=1e0)
learning_rate       = Float(    "learning_rate",        (1e-2, 5e-1), default=1e-1)
configuration_space.add_hyperparameters([objective, num_parallel_tree, n_estimators, max_depth, subsample, learning_rate])

metrics_df, organism_metrics_df, overall_metrics_df = nested_cross_validate_model_sklearn( X=X, ys=ys, labels=targets, classifier=classifier,
                                                                                           configuration_space=configuration_space, n_trials=n_trials,
                                                                                           name=name, algorithm_name=algorithm_name, outdir=outdir,
                                                                                           fold=StratifiedKFold(n_splits=outer_fold), inner_fold=inner_fold, verbosity=verbosity)

plot_metrics_df(metrics_df, organism_metrics_df, overall_metrics_df, algorithm_name, outdir, show=False)



configuration_space = ConfigurationSpace()
max_layers = 8
dropout_in = Float("dropout_in", (0.0, 0.9), default=0.5)
n_layers = Integer("n_layers", (1, max_layers), default=1)
n_neurons = []
activations = []
dropouts = []
for i in range(max_layers):
    n_neurons.append(Integer(f"n_neurons_{i}", (10, 10000), log=True, default=100))
    activations.append( Categorical(f"activation_{i}", ["tanh", "relu", "leakyrelu", "sigmoid"], default="relu") )
    dropouts.append( Categorical(f"dropout_{i}", [True, False], default=True) )
solver = Constant("solver", "nadam")
learning_rate = Float("learning_rate", (1e-4, 1e-1), log=True, default=1e-2)

hyperparameters = n_neurons + activations + dropouts + [dropout_in, n_layers, solver, learning_rate]
configuration_space.add_hyperparameters( hyperparameters )

outdir = Path(os.path.normpath("/mnt/d/runs/ML/NN/smac"))

metrics_df, organism_metrics_df, overall_metrics_df = nested_cross_validate_model_keras( X, ys, targets, configuration_space=configuration_space, n_trials=10, classes=1,
                                                                                         fold=StratifiedKFold(n_splits=5), patience=1000, epochs=10000, outdir=outdir,
                                                                                         verbosity=verbosity )

plot_metrics_df(metrics_df, organism_metrics_df, overall_metrics_df, algorithm_name, outdir, show=False)