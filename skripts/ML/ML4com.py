# imports
import os
import time
import warnings
import math
import itertools
from tqdm import tqdm
from typing import List
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.utils.random import sample_without_replacement
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn import tree
from sklearn import svm
import seaborn as sns
import matplotlib.pyplot as plt



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
    for y in ys.transpose():

        # Predict the test set labels
        y_pred = cross_val_predict(model, X.transpose(), y, cv=n_fold)

        confusion_matrices.append(confusion_matrix(y, y_pred))
        accuracies.append(accuracy_score(y,y_pred))
    return (accuracies, confusion_matrices)


def grid_search_params_cv_model(classifier, param_grid, X, ys, targets, n_splits:int=5, n_jobs:int=1):
    grids = {}
    for i, y in enumerate(ys.transpose()):
        # Model definition
        cv = StratifiedShuffleSplit(n_splits=n_splits, test_size=1/n_splits, random_state=42)
        grid = GridSearchCV(classifier(), param_grid=param_grid, cv=cv, n_jobs=n_jobs, verbose=2)
        grid.fit(X.transpose(), y)
        grids[targets[i].item()] = grid
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
            
            param_dict_str = "_".join(["-".join([str(k), str(v)]) for k,v in param_dict.items()])
            name = f"{param_dict_str}_{suffix}"
            plot_cv_confmat(ys=ys, target_labels=target_labels, accuracies=accuracies, confusion_matrices=confusion_matrices, outdir=outdir, name=name)

    results.to_csv(os.path.join(outdir, f"accuracies_{suffix}.tsv"), sep="\t")
    return results



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