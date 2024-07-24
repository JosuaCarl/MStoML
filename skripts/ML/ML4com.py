# imports
import os
import gc
import time
import warnings
import itertools
from typing import Union
from tqdm import tqdm
from typing import List
from pathlib import Path
import numpy as np
import pandas as pd
import pickle

import sklearn
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_predict
from sklearn.utils.random import sample_without_replacement
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn import tree

import seaborn as sns
import matplotlib.pyplot as plt

import mlflow
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['MALLOC_TRIM_THRESHOLD_'] = '0'

import tensorflow as tf
import keras
from keras import layers, activations, backend

from ConfigSpace import Categorical, Configuration, ConfigurationSpace, EqualsCondition, Float, InCondition, Integer, Constant, ForbiddenAndConjunction, ForbiddenEqualsClause
from smac import HyperparameterOptimizationFacade, MultiFidelityFacade, Scenario
from smac.intensifier.hyperband import Hyperband



## Helpers
def dict_permutations(dictionary:dict) -> List[dict]:
  """
  Combine all value combinations in a dictionary into a list of dictionaries.
  """
  keys, values = zip(*dictionary.items())
  return [dict(zip(keys, v)) for v in itertools.product(*values)]


def join_df_metNames(df, grouper="peakID", include_mass=False):
    """
    Sets common index for combination of positively and negatively charged dataframes along their metabolite Names
    """
    mass_name = "ref_mass" if "ref_mass" in df.columns else "mass"
    data_cols = [col for col in df.columns[df.dtypes == "float"] if col not in [mass_name, "kegg", "kegg_id", "dmz"]]
    cols = ["metNames"] + [f"MS{i+1}" for i in range(len(data_cols))]
    if include_mass:
        cols = cols + [mass_name]
    comb = pd.DataFrame(columns=cols)
    
    grouper = mass_name if "mass" in grouper else grouper
    for g in df[grouper].unique():
        comb_met_name = ""
        grouped_rows = df.loc[df[grouper] == g]
        for i, row in grouped_rows.iterrows():
            comb_met_name += row["MetName"] + "\n"
            ref_mass = row[mass_name]
        if include_mass:
            comb.loc[len(comb.index)] = [comb_met_name[:-2], ref_mass] + list(grouped_rows.iloc[0][data_cols])
        else:
            comb.loc[len(comb.index)] = [comb_met_name[:-2]] + list(grouped_rows.iloc[0][data_cols])
    comb = comb.set_index('metNames')
    return comb


def intersect_impute_on_left(df_base:pd.DataFrame, df_right:pd.DataFrame, imputation:str="zero"):
    """
    Use all indices from the left (df_base) DataFrame and fill it with the intersection of df_right.
    Indices without match are imputed by zero, mean or via kNN, whereas k can be specified as a number.
    The default is 5.
    """
    df_merged = pd.merge(df_base, df_right, left_index=True, right_index=True, how="left")
    df_merged = df_merged.loc[:, ~df_merged.columns.str.endswith('_x')]
    df_merged.columns = [col.replace("_y", "") for col in df_merged.columns]
    df_merged = df_merged[df_right.columns]
    if imputation == "zero":
        df_merged = df_merged.fillna(0.0)
    elif imputation == "mean":
        df_merged = df_merged.fillna(np.mean(df_merged))
    elif imputation.endswith("NN"):
        k = 5 if imputation[0] == "k" else int(imputation[0])
        imputer = KNNImputer(n_neighbors=k, keep_empty_features=True)
        df_merged.values = imputer.fit_transform(df_merged)
    return df_merged



## SKLearn
def individual_layers_to_tuple(config) -> dict:
    config = dict(config)
    hidden_layer_sizes = tuple([config.pop(k) for k in list(config.keys()) if k.startswith("n_neurons")])
    if hidden_layer_sizes:
        config["hidden_layer_sizes"] = hidden_layer_sizes
    return config


class SKL_Classifier:
    """
    Representation of Scikit-learn classifier for SMAC3
    """
    def __init__(self, X, ys, cv, configuration_space:ConfigurationSpace, classifier, n_trials):
        self.X = X
        self.ys = ys
        self.cv = cv
        self.configuration_space = configuration_space
        self.classifier = classifier
        self.count = 0
        self.progress_bar = tqdm(total=n_trials)

    def train(self, config: Configuration, seed:int=0) -> np.float64:
        config = individual_layers_to_tuple(config)
        if "hidden_layer_sizes" in config:
            self.progress_bar.set_postfix_str(f'Connection size: {np.prod(config["hidden_layer_sizes"])}')

        with mlflow.start_run(run_name=f"run_{self.count}", nested=True):
            mlflow.set_tag("test_identifier", f"child_{self.count}")
            if isinstance(self.cv, int):
                splitter = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=seed)
            else:
                splitter = self.cv
            scores = []
            for train, test in splitter.split(self.X, self.ys):
                model = self.classifier(**config)
                model.fit(self.X.iloc[train].values, self.ys.iloc[train].values)
                y_pred = model.predict(self.X.iloc[test].values)
                scores.append( np.mean( y_pred == self.ys.iloc[test].values) )
            score = np.mean( scores )
            mlflow.log_params( config )
            mlflow.log_metrics( {"accuracy": score} )
            self.count += 1

        self.progress_bar.update(1)
        return 1.0 - score


def tune_classifier(X, y, classifier, cv, configuration_space, n_workers, n_trials, name, algorithm_name, outdir, verbosity):
    """
    Perform hyperparameter tuning on an Sklearn classifier.
    """
    classifier = SKL_Classifier(X, y, cv=cv, configuration_space=configuration_space, classifier=classifier, n_trials=n_trials)

    scenario = Scenario( classifier.configuration_space, deterministic=True, 
                         n_workers=n_workers, n_trials=n_trials,
                         walltime_limit=np.inf, cputime_limit=np.inf, trial_memory_limit=None,
                         output_directory=outdir )

    facade = HyperparameterOptimizationFacade(scenario, classifier.train, overwrite=True, logging_level=30-verbosity*10)

    mlflow.set_tracking_uri(Path(os.path.join(outdir, "mlruns")))
    mlflow.set_experiment(f"{name}_{algorithm_name}")
    with mlflow.start_run(run_name=f"{name}_{algorithm_name}"):
        mlflow.set_tag("test_identifier", "parent")
        incumbent = facade.optimize()

    return incumbent


def extract_metrics(true_labels, prediction, scoring, run_label=None, cv_i=None,
                    metrics_df:pd.DataFrame=pd.DataFrame(columns=["Run", "Cross-Validation run", "Accuracy", "AUC", "TPR", "FPR", "Threshold", "Conf_Mat"])):
    """
    Extract meaningful metrics to score a model according to its label prediction in comparison to true labels.
    """
    fpr, tpr, threshold = roc_curve(true_labels,  scoring)
    auc = roc_auc_score(true_labels,  scoring)
    conf_mat = confusion_matrix(true_labels,  prediction)
    accuracy = accuracy_score(true_labels,  prediction)
    
    if cv_i and run_label:
        metrics_df.loc[len(metrics_df)] = [run_label, cv_i, accuracy, auc, tpr, fpr, threshold, conf_mat]
    elif run_label:
        metrics_df.loc[len(metrics_df)] = [run_label, accuracy, auc, tpr, fpr, threshold, conf_mat]
    else:
        metrics_df.loc[len(metrics_df)] = [accuracy, auc, tpr, fpr, threshold, conf_mat]

    return metrics_df


def extract_best_hyperparameters_from_incumbent(incumbent, configuration_space):
    if not incumbent:
        best_hp = configuration_space.get_default_configuration()
    elif isinstance(incumbent, list):
        best_hp = incumbent[0]
    else: 
        best_hp = incumbent
    return best_hp



def nested_cross_validate_model_sklearn(X, ys, labels, classifier, configuration_space, n_trials,
                                        name, algorithm_name, outdir, fold:Union[KFold, StratifiedKFold]=KFold(),
                                        inner_fold:int=3, n_workers:int=1, verbosity:int=0):
    """
    Cross-validate a model against the given hyperparameters for all organisms
    """
    metrics_df = pd.DataFrame(columns=["Organism", "Cross-Validation run", "Accuracy", "AUC", "TPR", "FPR", "Threshold", "Conf_Mat"])
    organism_metrics_df = pd.DataFrame(columns=["Organism", "Accuracy", "AUC", "TPR", "FPR", "Threshold", "Conf_Mat"])
    overall_metrics_df = pd.DataFrame(columns=["Accuracy", "AUC", "TPR", "FPR", "Threshold", "Conf_Mat"])

    all_predictions = np.ndarray((0))
    all_scorings = np.ndarray((0))
    ys_deshuffeld = np.ndarray((0))

    # Iterate over all organisms for binary distinction
    for i, y in enumerate(tqdm(ys.columns)):
        y = ys[y]
        predictions = np.ndarray((0))
        scorings = np.ndarray((0))
        y_deshuffled = np.ndarray((0))

        # Outer Loop
        for cv_i, (train_index, val_index) in enumerate(tqdm(fold.split(X, y))):
            training_data = X.iloc[train_index]
            training_labels = y.iloc[train_index]
            validation_data = X.iloc[val_index]
            validation_labels = y.iloc[val_index]

            # Hyperparameter Tuning with inner CV loop
            incumbent = tune_classifier(training_data, training_labels, classifier, inner_fold, configuration_space,
                                        n_workers, n_trials, name, algorithm_name, outdir, verbosity)
            
            # Model definition and fitting
            best_hp = extract_best_hyperparameters_from_incumbent(incumbent=incumbent, configuration_space=configuration_space)
            best_hp = individual_layers_to_tuple(best_hp)
            model = classifier(**best_hp)		# Ensures model resetting for each cross-validation

            model.fit(np.array(training_data), np.array(training_labels))

            # Prediction and scoring
            prediction = model.predict( np.array(validation_data) )
            if hasattr(model, "decision_function"):
                scoring = model.decision_function( np.array(validation_data) )
            elif hasattr(model, "predict_proba"):
                scoring = model.predict_proba( np.array(validation_data) )[::,1]
            else:
                scoring = prediction
            
            metrics_df = extract_metrics(validation_labels, prediction, scoring, labels[i], cv_i+1, metrics_df)
			
            if verbosity != 0:
                if hasattr(model, "evaluate"):
                    model.evaluate(validation_data,  validation_labels, verbose=verbosity)
                if hasattr(model, "score"):
                    print(f"Mean accuracy: {model.score(validation_data,  validation_labels)}")

            predictions = np.append(predictions, prediction)
            scorings = np.append(scorings, scoring)
            y_deshuffled = np.append(y_deshuffled, validation_labels)

        organism_metrics_df = extract_metrics(y_deshuffled, predictions, scorings, labels[i], metrics_df=organism_metrics_df)
        all_predictions = np.append(all_predictions, predictions)
        all_scorings = np.append(all_scorings, scorings)
        ys_deshuffeld = np.append(ys_deshuffeld, y_deshuffled)

    overall_metrics_df = extract_metrics(ys_deshuffeld, all_predictions, all_scorings, metrics_df=overall_metrics_df)

    # Saving of results
    metrics_df.to_csv(os.path.join(outdir, f"{algorithm_name}_metrics.tsv"), sep="\t")
    organism_metrics_df.to_csv(os.path.join(outdir, f"{algorithm_name}_organism_metrics.tsv"), sep="\t")
    overall_metrics_df.to_csv(os.path.join(outdir, f"{algorithm_name}_overall_metrics.tsv"), sep="\t")

    return (metrics_df, organism_metrics_df, overall_metrics_df)


def tune_train_model_sklearn( X, ys, labels, classifier, configuration_space, n_workers, n_trials,
                                source:str, name, algorithm_name, outdir, fold:Union[KFold, StratifiedKFold]=KFold(),
                                verbosity=0 ):
    """
    Tune and train a model against the given hyperparameters for all organisms
    """
    # Iterate over all organisms for binary distinction
    for i, org in enumerate(tqdm(ys.columns)):
        y = ys[org]

        incumbent = tune_classifier(X, y, classifier, fold, configuration_space, n_workers, n_trials,
                                    name, algorithm_name, outdir, verbosity)
        
        # Model definition and fitting
        best_hp = extract_best_hyperparameters_from_incumbent(incumbent=incumbent, configuration_space=configuration_space)

        best_hp = individual_layers_to_tuple(best_hp)
        model = classifier(**best_hp)		# Ensures model resetting for each cross-validation

        model.fit(np.array(X), np.array(y))
        with open(os.path.join(outdir, f'model_{algorithm_name}_{labels[i]}_{source}.pkl'), 'wb') as f:
            pickle.dump(model ,f)


def evaluate_model_sklearn( X, ys, labels, indir, data_source, algorithm_name, outdir, verbosity=0 ):
    """
    Evaluate a model against the given hyperparameters for all organisms and save the resulting metrics and feature importances.
    """
    eval_metrics_df = pd.DataFrame(columns=["Organism", "Accuracy", "AUC", "TPR", "FPR", "Threshold", "Conf_Mat"])
    feature_importances = {}

    for i, org in enumerate(tqdm(ys.columns)):
        y = ys[org]

        with open(os.path.join(indir, f'model_{algorithm_name}_{labels[i]}.pkl'), 'rb') as f:
            model = pickle.load(f)

        prediction = model.predict( np.array(X) )
        if hasattr(model, "decision_function"):
            scoring = model.decision_function( np.array(X) )
        elif hasattr(model, "predict_proba"):
            scoring = model.predict_proba( np.array(X) )[::,1]
        else:
            scoring = prediction
        
        eval_metrics_df = extract_metrics(y, prediction, scoring, labels[i], 0, eval_metrics_df)

        model.fit(X.values, y)
        if hasattr(model, "feature_importances_"):
            feature_importances[labels[i]] = model.feature_importances_

    feat_imp_df = pd.DataFrame(feature_importances, index=X.columns)
    feat_imp_df.to_csv(os.path.join(outdir, f"feature_importance_{algorithm_name}_{data_source}.tsv"), sep="\t")

    eval_metrics_df.to_csv(os.path.join(outdir, f"{algorithm_name}_{data_source}_eval_metrics.tsv"), sep="\t")



# PLOTTING
def plot_cv_confmat(ys, target_labels, accuracies, confusion_matrices, outdir, name):
    """
    Plot heatmap of confusion matrix
    """
    warnings.filterwarnings("ignore", message="This figure includes Axes that are not compatible with tight_layout, so results might be incorrect*")
    fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(16, 8))
    cbar_ax = fig.add_axes([.91, .3, .03, .4])          # type: ignore
    for i, ax in enumerate(axs.flat):
        sns.heatmap(confusion_matrices[i], 
                    vmin=0, vmax=len(ys), annot=True, ax=ax, 
                    cbar=i == 0, cbar_ax=None if i else cbar_ax)
        ax.set_title(f'{target_labels[i]}, Accuracy: {round(accuracies[i], 5)}')
        ax.axis('off')
    plt.title(f"Overall accuracy: {round(np.mean(accuracies), 3)}")
    fig.tight_layout(rect=[0, 0, .9, 1])                # type: ignore
    plt.savefig(os.path.join(outdir, f"{name}.png"))
    plt.close()
    

def plot_decision_trees(model, feature_names, class_names, outdir, name):
    """
    Plot decision tree of a model.
    """
    fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=900)
    tree.plot_tree(model,
                feature_names = feature_names, 
                class_names = class_names,
                filled = True)
    plt.savefig(os.path.join(outdir, f"{name}.png"))
    plt.close()


def plot_metrics_df(metrics_df, organism_metrics_df, overall_metrics_df, algorithm_name, outdir, show=False):
    """
    Plot the extracted metrics as a heatmap and ROC AUC curve
    """
    ax = sns.heatmap(metrics_df.pivot(index="Organism", columns="Cross-Validation run", values="Accuracy"),
                    vmin=0, vmax=1.0, annot=True, cmap=sns.diverging_palette(328.87,  221.63, center="light", as_cmap=True))
    fig = ax.get_figure()
    fig.savefig(os.path.join(outdir, f"{algorithm_name}_heatmap_accuracies.png"), bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

    ax = sns.lineplot( overall_metrics_df.explode(["TPR", "FPR"]), x="FPR", y="TPR", hue="AUC")
    ax = sns.lineplot( organism_metrics_df.explode(["TPR", "FPR"]), x="FPR", y="TPR", hue="Organism", alpha=0.5, ax=ax)
    ax.set_title("AUC")
    leg = ax.axes.get_legend()
    leg.set_title("Organism (AUC)")
    for t, l in zip(leg.texts, [f'{row["Organism"]} (AUC={str(row["AUC"])})' for i, row in organism_metrics_df.iterrows()]+ [f"Overall (AUC={overall_metrics_df.loc[0, 'AUC']})"]):
        t.set_text(l)
    fig = ax.get_figure()
    fig.savefig(os.path.join(outdir, f"{algorithm_name}_AUC.png"), bbox_inches="tight")
    if show:
        plt.show()
    plt.close()