#!/usr/bin/env python3
"""
ML4com runscript through shell commands. The parameter boundaries for the hyperparameters are defined here.
"""

# imports
import sys
sys.path.append( '..' )

from FIA.FIA import *
from ML.ML4com import *
from VAE.vae import *

import warnings
warnings.simplefilter(action='ignore', category=sklearn.exceptions.UndefinedMetricWarning)


def main(args):
    """Main method for executing runscript

    :param args: arguments for runscript, see ML4com_run.py --help for more information
    """    
    algorithms = args.algorithms if args.algorithms else None
    task = args.task if args.task else None

    n_workers = args.workers if args.workers else 1
    n_trials = args.trials
    inner_fold = args.inner_fold
    outer_fold = args.outer_fold
    verbosity = args.verbosity

    start_dir = args.start_dir
    source = args.source
    model_source = args.model_source if args.model_source else None
    sample = args.sample

    backend_name = args.backend
    computation = args.computation
    name = args.name if args.name else None
    file_name = args.filename
    model_filename = args.model_filename if args.model_filename else None

    
    orig_dir = os.path.normpath(os.path.join(os.getcwd(), f"{start_dir}/data/{sample}"))
    data_dir  = os.path.normpath(os.path.join(os.getcwd(), f"{start_dir}/runs/FIA/{sample}"))


    print("Start:")
    strains = pd.read_csv( os.path.join(orig_dir, "strains.tsv"), sep="\t")
      
    if source == "latent":
        run_dir = os.path.normpath(os.path.join(os.getcwd(), f"{start_dir}/runs/ML/latent"))

        project = f"vae_{backend_name}_{computation}_{name}"
        outdir = Path( os.path.normpath( os.path.join(run_dir, f"{sample}_{name}")) )
        if not os.path.isdir(outdir):
            os.mkdir(outdir)

        ys = pd.read_csv( os.path.join(orig_dir, "comb_one_hot_alternating.tsv"), sep="\t")
        X = pd.read_csv(f"{start_dir}/runs/VAE/results/{sample}/encoded_mu_{name}.tsv", index_col=0, sep="\t")

    elif source == "annotated":
        run_dir = os.path.normpath(os.path.join(os.getcwd(), f"{start_dir}/runs/ML/annot"))
        
        met_raw_pos = pd.read_excel( os.path.join( orig_dir, file_name ), sheet_name="pos" )
        met_raw_neg = pd.read_excel( os.path.join( orig_dir, file_name ), sheet_name="neg" )
        met_raw_comb = pd.concat( [ total_ion_count_normalization( join_df_metNames(met_raw_pos, grouper="mass") ),
                                    total_ion_count_normalization( join_df_metNames(met_raw_neg, grouper="mass") ) ] )

        outdir = Path( os.path.normpath( os.path.join( run_dir, sample) ) )
        if not os.path.isdir(outdir):
            os.mkdir(outdir)

        ys = pd.read_csv( os.path.join(orig_dir, "comb_one_hot.tsv"), sep="\t")
        X = met_raw_comb.transpose()

    labels = strains["Organism"].values

    print(f"Shape of input data: {X.shape}")

    algorithms_configspaces = {}
    
    # KNN
    from sklearn.neighbors import KNeighborsClassifier
    configuration_space = ConfigurationSpace()
    hyperparameters = [ 
                        Integer("n_neighbors",   (2,50), default=5),
                        Categorical("weights",    ["uniform", "distance"], default="uniform"),
                        Integer("leaf_size",      (10, 100), log=True, default=30),
                        Integer("p",              (1, 2), default=2),
                        Categorical("metric",     ["minkowski", "cosine"], default="minkowski")
                    ]
    configuration_space.add_hyperparameters( hyperparameters )
    conditions = [
                    InCondition(hyperparameters[3], hyperparameters[4], ["minkowski"])
                ]

    algorithms_configspaces["K-neighbours classifier"] = {"classifier": KNeighborsClassifier, "configuration_space": configuration_space}


    # SVC
    from sklearn.svm import SVC 
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

    algorithms_configspaces["Support-vector classifier"] = {"classifier": SVC, "configuration_space": configuration_space}


    # Gaussian Process
    from sklearn.gaussian_process import GaussianProcessClassifier
    configuration_space = ConfigurationSpace()
    hyperparameters = [ 
                        Constant("random_state", 42),
                        Integer("max_iter_predict",   (10, 1000), log=True, default=100),
                    ]
    configuration_space.add_hyperparameters( hyperparameters )

    algorithms_configspaces["Gaussian process classifier"] = {"classifier": GaussianProcessClassifier, "configuration_space": configuration_space}



    # Naive Bayes
    from sklearn.naive_bayes import GaussianNB
    configuration_space = ConfigurationSpace()
    hyperparameters = [ 
                        Constant("var_smoothing",   1e-9),
                    ]
    configuration_space.add_hyperparameters( hyperparameters )

    algorithms_configspaces["Gaussian Naive-Bayes"] = {"classifier": GaussianNB, "configuration_space": configuration_space}



    # Decision Tree
    from sklearn.tree import DecisionTreeClassifier
    configuration_space = ConfigurationSpace()
    hyperparameters = [
        Constant( "random_state",          42),
        Float("ccp_alpha", (1e-3, 1e0), log=True, default=0.01)
    ]
    configuration_space.add_hyperparameters(hyperparameters)

    algorithms_configspaces["Decision tree"] = {"classifier": DecisionTreeClassifier, "configuration_space": configuration_space}


    # Random Forest
    from sklearn.ensemble import RandomForestClassifier
    configuration_space = ConfigurationSpace()
    hyperparameters = [
        Constant( "random_state",          42),
        Float("ccp_alpha", (1e-3, 1e-1), log=True, default=0.01),
        Integer("n_estimators", (10,1000), log=True, default=100),
        Integer("max_depth", (5, 100), default=20),
    ]
    configuration_space.add_hyperparameters(hyperparameters)

    algorithms_configspaces["Random forest"] = {"classifier": RandomForestClassifier, "configuration_space": configuration_space}


    # Extreme Gradient Boosting
    from xgboost import XGBClassifier
    configuration_space = ConfigurationSpace()
    hyperparameters = [
        Constant( "random_state",          42),
        Constant( "device",               "cuda"),
        Constant( "objective",            "binary:logistic"),
        Integer( "num_parallel_tree",     (1, 100), log=True, default=4),
        Integer(  "n_estimators",         (10,1000), log=True, default=100),
        Integer(  "max_depth",            (1, 100), default=20),
        Float(    "subsample",            (1e-1, 1e0), default=1e0),
        Float(    "learning_rate",        (1e-2, 5e-1), default=1e-1),
    ]
    configuration_space.add_hyperparameters(hyperparameters)

    algorithms_configspaces["Extreme gradient boosting RF"] = {"classifier": XGBClassifier, "configuration_space": configuration_space}


    # Multi-Layer-Perceptron
    from sklearn.neural_network import MLPClassifier
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    configuration_space = ConfigurationSpace()
    n_neurons = []
    for i in range(5):
        n_neurons.append(Integer(f"n_neurons_{i}", (8, 1024), log=True, default=128))
    hyperparameters = [
        Categorical(    "activation",           ["tanh", "relu", "logistic"], default="relu"),
        Constant(       "solver",               "adam"),
        Float(          "alpha",                (1e-5, 1e-1), log=True, default=1e-2),
        Categorical(    "learning_rate",        ["constant", "adaptive"], default="constant"),
        Constant(       "learning_rate_init",   0.001),
        Constant(       "random_state",         42),
        Float(          "momentum",             (0.9, 0.99), default=0.9),
        Categorical(    "nesterovs_momentum",   [True], default=True),
        Constant(       "validation_fraction",  0.2),
        Float(          "beta_1",               (0.9, 0.99), default=0.9),
        Float(          "beta_2",               (0.99, 0.9999), default=0.999),
        Constant(       "epsilon",              1e-12),
    ] + n_neurons

    configuration_space.add_hyperparameters( hyperparameters )

    algorithms_configspaces["Neural Network (MLP) SK-learn"] = {"classifier": MLPClassifier, "configuration_space": configuration_space}



    # LDA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    configuration_space = ConfigurationSpace()

    hyperparameters = [ 
                        Categorical( "solver",      ["svd", "lsqr", "eigen"], default="svd"),
                        Categorical("shrinkage",    ["auto"], default="auto"),
                        Float("tol",                (1e-6,1e-2), log=True, default=1e-4)
                    ]
    conditions = [
                    InCondition(hyperparameters[1], hyperparameters[0], ["lsqr", "eigen"])
                ]
    configuration_space.add_hyperparameters(hyperparameters)
    configuration_space.add_conditions(conditions)
    

    algorithms_configspaces["Linear Discriminant Analysis"] = {"classifier": LinearDiscriminantAnalysis, "configuration_space": configuration_space}



    # Logistic Regression
    from sklearn.linear_model import LogisticRegression
    configuration_space = ConfigurationSpace()

    hyperparameters = [ 
                        Constant("random_state",        42),
                        Constant("max_iter",            100),
                        Categorical("penalty",          ["l1", "l2", "elasticnet"], default="l2"),
                        Categorical( "solver",          ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"], default="lbfgs"),
                        Float("tol",                    (1e-6,1e-2), log=True, default=1e-4),
                        Float("C",                      (1e-3,1e3), log=True, default=1.0),
                        Float("intercept_scaling",      (1e-3,1e3), log=True, default=1.0),
                        Float("l1_ratio",               (0.0, 1.0), default=0.5)
                    ]
    conditions = [
                    InCondition(hyperparameters[7], hyperparameters[2], ["elasticnet"]),
                    InCondition(hyperparameters[2], hyperparameters[3], ["saga", "liblinear"])           
                ]
    clauses = [
        ForbiddenAndConjunction(
                                    ForbiddenEqualsClause(hyperparameters[3], "liblinear"),
                                    ForbiddenEqualsClause(hyperparameters[2], "elasticnet")
                                )
            ]
    configuration_space.add_hyperparameters( hyperparameters )
    configuration_space.add_conditions( conditions )
    configuration_space.add_forbidden_clauses( clauses )

    algorithms_configspaces["Logistic Regression"] = {"classifier": LogisticRegression, "configuration_space": configuration_space}
    
    
    if algorithms and "all" not in algorithms:
        algorithms_configspaces_acc = {}
        for algorithm in algorithms:
            if algorithm in algorithms_configspaces:
                algorithms_configspaces_acc[algorithm] = algorithms_configspaces[algorithm]
            else:
                raise( ValueError(f"-a/--algorithm {algorithm} is unknown. Choose one of {list(algorithms_configspaces.keys())}"))
        algorithms_configspaces = algorithms_configspaces_acc

    print(f"Starting at: {datetime.datetime.now()}")
    if task == "train":
        print("Training:")
        for algorithm_name in tqdm(list(algorithms_configspaces.keys())):
            print(f"\n{algorithm_name}:")
            tune_train_model_sklearn( X=X, ys=ys, labels=labels, classifier=algorithms_configspaces[algorithm_name]["classifier"],
                                        configuration_space=algorithms_configspaces[algorithm_name]["configuration_space"],
                                        n_workers=n_workers, n_trials=n_trials, source=source, name=name, algorithm_name=algorithm_name, outdir=outdir,
                                        fold=StratifiedKFold(n_splits=outer_fold), verbosity=verbosity )
    elif task == "evaluate":
        print("Evaluating:")
        if source == "latent":
            indir = Path( os.path.normpath( os.path.join(run_dir, f"{model_source}_{name}")) )
        elif source == "annotated":
            indir = Path( os.path.normpath( os.path.join(run_dir, model_source)) )

            # Makes transfer of data along annotated Data possible
            orig_dir_2 = os.path.normpath(os.path.join(os.getcwd(), f"{start_dir}/data/{model_source}"))
            met_raw_pos_2 = pd.read_excel( os.path.join( orig_dir_2, model_filename ), sheet_name="pos" )
            met_raw_neg_2 = pd.read_excel( os.path.join( orig_dir_2, model_filename ), sheet_name="neg" )
            met_raw_comb_2 = pd.concat( [ total_ion_count_normalization( join_df_metNames(met_raw_pos_2, grouper="mass") ),
                                          total_ion_count_normalization( join_df_metNames(met_raw_neg_2, grouper="mass") ) ] )
            X = intersect_impute_on_left(met_raw_comb_2, met_raw_comb, imputation="zero").transpose()
    	
        print(f"Evalutation of {os.path.basename(outdir)} on models trained with {os.path.basename(indir)}. Saved in {indir} as algorithm_{sample}_eval_metrics.tsv")
        for algorithm_name in tqdm(list(algorithms_configspaces.keys())):
            evaluate_model_sklearn( X=X, ys=ys, labels=labels, indir=indir, data_source=sample, algorithm_name=algorithm_name, outdir=outdir,
                                    verbosity=verbosity )
    else:
        print("Tuning:")
        for algorithm_name, alg_info in tqdm(algorithms_configspaces.items()):
            print(f"\n{algorithm_name}:")
            metrics_df, organism_metrics_df, overall_metrics_df = nested_cross_validate_model_sklearn( X=X, ys=ys,
                                                                                                       labels=labels,
                                                                                                       classifier=alg_info["classifier"],
                                                                                                       configuration_space=alg_info["configuration_space"],
                                                                                                       n_trials=n_trials,
                                                                                                       name=name, algorithm_name=algorithm_name,
                                                                                                       outdir=outdir,
                                                                                                       fold=StratifiedKFold(n_splits=outer_fold),
                                                                                                       inner_fold=inner_fold,
                                                                                                       n_workers=n_workers,
                                                                                                       verbosity=verbosity)

            plot_metrics_df(metrics_df, organism_metrics_df, overall_metrics_df, algorithm_name, outdir, show=False)

    print(f"FInished at {datetime.datetime.now()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='ML4com_run',
                                     description='Try different algorithms in nested-cv run.')
    
    parser.add_argument('-s', '--source', required=True)
    parser.add_argument('-ms', '--model_source', required=False)
    parser.add_argument('-st', '--start_dir', required=True)
    parser.add_argument('-sam', '--sample', required=True)
    parser.add_argument('-fn', '--filename', required=True)
    parser.add_argument('-mfn', '--model_filename', required=False)
    

    parser.add_argument('-w', '--workers', type=int, required=False)
    parser.add_argument('-t', '--trials', type=int, required=True)
    parser.add_argument('-if', '--inner_fold', type=int, required=True)
    parser.add_argument('-of', '--outer_fold', type=int, required=False)

    parser.add_argument('-b', '--backend',  required=True)
    parser.add_argument('-c', '--computation', required=True)
    parser.add_argument('-n', '--name', required=False)

    parser.add_argument('-a', '--algorithms', nargs="+", required=False)
    parser.add_argument('-ta', '--task', required=False)

    parser.add_argument('-v', '--verbosity', type=int, required=False)
    args = parser.parse_args()

    main(args)
