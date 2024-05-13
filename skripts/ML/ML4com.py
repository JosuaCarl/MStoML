# imports
import os
import time
import warnings
import math
import itertools
from typing import Union
from tqdm import tqdm
from typing import List
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_predict
from sklearn.utils.random import sample_without_replacement
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn import tree
from sklearn import svm
import seaborn as sns
import matplotlib.pyplot as plt
from ConfigSpace import Configuration, ConfigurationSpace
import mlflow
from pathlib import Path


# Combination of pos/neg 
def join_df_metNames(df, include_mass=False):
    """
    Combines positively and negatively charged dataframes along their metabolite Names
    """
    cols = ["metNames", "ref_mass"] + [f"MS{i+1}" for i in range(len(df.columns) - 6)]
    comb = pd.DataFrame(columns=cols)
    for pid in df["peakID"].unique():
        comb_met_name = ""
        for i, row in df.loc[df["peakID"] == pid].iterrows():
            comb_met_name += row["MetName"] + "\n"
            ref_mass = row["ref_mass"]
        if include_mass:
            comb.loc[len(comb.index)] = [comb_met_name[:-2], ref_mass] + list(df.loc[df["peakID"] == pid].iloc[0, 6:])
        else:
            comb.loc[len(comb.index)] = [comb_met_name[:-2]] + list(df.loc[df["peakID"] == pid].iloc[0, 6:])
    comb = comb.set_index('metNames')
    return comb

# Helpers
def dict_permutations(dictionary:dict) -> List[dict]:
  keys, values = zip(*dictionary.items())
  return [dict(zip(keys, v)) for v in itertools.product(*values)]

# SKLEARN
def mult_cv_model(model, X, ys, n_fold):
    """
    Performs model training and cross-validation over each column
    """
    # Matrices for report
    accuracies = []
    confusion_matrices = []
    
    # Perform cross-validation for each strain separately
    kf = KFold(n_splits=n_fold)
    for y in ys.transpose():
        validation_preds = np.array([])
        accs = []
        for train_index, val_index in kf.split(X, y):
            model_test = model
            training_data = X[train_index]
            training_labels = y[train_index]
            validation_data = X[val_index]
            validation_labels = y[val_index]

            model_test.fit(training_data, training_labels)
            validation_pred = model_test.predict(validation_data)
            accs.append(accuracy_score(validation_labels, validation_pred))
            validation_preds = np.append(validation_preds, validation_pred)
        accuracies.append(accs)
        confusion_matrices.append(confusion_matrix(y, validation_preds))

    return (accuracies, confusion_matrices)


def grid_search_params_cv_model(classifier, param_grid, X, ys, targets, n_splits:int=5, n_jobs:int=1):
    grids = {}
    for i, y in enumerate(ys.transpose()):
        # Model definition
        cv = StratifiedShuffleSplit(n_splits=n_splits, test_size=1/n_splits, random_state=42)
        grid = GridSearchCV(classifier(), param_grid=param_grid, cv=cv, n_jobs=n_jobs, verbose=2)
        grid.fit(X, y)
        grids[targets[i]] = grid
    return grids


def train_cv_model(classifier, param_grid, X, ys, target_labels, outdir:str, suffix:str="", n_fold:int=5):
    # Evaluation data
    results = pd.DataFrame(columns=["model_nr", "parameters", "target", "accuracy"])

    model_count = 1
    for i, grid in enumerate(param_grid):
        print(f"Parameter combinations {i+1}:")
        for param_dict in tqdm(dict_permutations(grid)):
            # Model definition
            model = classifier(**param_dict)

            accuracies, confusion_matrices = mult_cv_model(model=model, X=np.array(X), ys=np.array(ys), n_fold=n_fold)
            for i in range(len(target_labels)):
                results.loc[len(results.index)] = [model_count, param_dict, target_labels[i], accuracies[i], ]

            model_count += 1
            
            param_dict_str = "_".join(["-".join([str(k), str(v)]) for k,v in param_dict.items()]).replace(":", "")
            name = f"{param_dict_str}_{suffix}"
            plot_cv_confmat(ys=ys, target_labels=target_labels, accuracies=np.mean(accuracies, axis=1), confusion_matrices=confusion_matrices, outdir=outdir, name=name)

    results.to_csv(os.path.join(outdir, f"accuracies_{suffix}.tsv"), sep="\t")
    return results


## SMAC
class SKL_Classifier:
    def __init__(self, X, ys, cv:int, configuration_space:ConfigurationSpace, classifier):
        self.X = X
        self.ys = ys
        self.cv = cv
        self.configuration_space = configuration_space
        self.classifier = classifier
        self.count = 0

    def train(self, config: Configuration, seed:int=0) -> np.float64:
        with mlflow.start_run(run_name=f"run_{self.count}", nested=True):
            mlflow.set_tag("test_identifier", f"child_{self.count}")
            splitter = KFold(n_splits=self.cv, shuffle=True, random_state=seed)
            scores = []
            for train, test in splitter.split(self.X, self.ys):
                model = self.classifier(**config, random_state=seed)
                model.fit(self.X.loc[train], self.ys.loc[train])
                y_pred = model.predict(self.X.loc[test])
                scores.append( np.mean( y_pred == self.ys.loc[test].values) )
            score = np.mean( scores )
            mlflow.log_params( config )
            mlflow.log_metrics( {"accuracy": score} )
            self.count += 1
        return 1.0 - score

def extract_metrics(true_labels, prediction, run_label, cv_i,
                    metrics_df:pd.DataFrame=pd.DataFrame(columns=["Run", "Cross-Validation run", "Accuracy", "AUC", "TPR", "FPR", "Threshold", "Conf_Mat"])):
    fpr, tpr, threshold = roc_curve(true_labels,  prediction)
    auc = roc_auc_score(true_labels,  prediction)
    conf_mat = confusion_matrix(true_labels,  prediction)
    accuracy = accuracy_score(true_labels,  prediction)

    metrics_df.loc[len(metrics_df)] = [run_label, cv_i, accuracy, auc, tpr, fpr, threshold, conf_mat]

    return metrics_df


def cross_validate_model_sklearn(model_in, X, ys, labels, config, fold:Union[KFold, StratifiedKFold]=KFold(), verbosity=0):
    """
    Cross-validate a model against the given hyperparameters for all organisms
    """
    metrics_df = pd.DataFrame(columns=["Organism", "Cross-Validation run", "Accuracy", "AUC", "TPR", "FPR", "Threshold", "Conf_Mat"])

    for i, y in enumerate(tqdm(ys.columns)):
        y = ys[y]
        for cv_i, (train_index, val_index) in enumerate(fold.split(X, y)):
            model = model_in(**config)		# Ensures model resetting for each cross-validation
            training_data = X.iloc[train_index]
            training_labels = y.iloc[train_index]
            validation_data = X.iloc[val_index]
            validation_labels = y.iloc[val_index]

            model.fit(np.array(training_data), np.array(training_labels))

            prediction = model.predict(np.array(validation_data))
            metrics_df = extract_metrics(validation_labels, prediction, labels[i], cv_i+1, metrics_df)
			
            if verbosity != 0:
                model.evaluate(validation_data,  validation_labels, verbose=verbosity) # type: ignore
    return metrics_df



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
    fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=900)
    tree.plot_tree(model,
                feature_names = feature_names, 
                class_names = class_names,
                filled = True)
    plt.savefig(os.path.join(outdir, f"{name}.png"))
    plt.close()