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

from ConfigSpace import Categorical, Configuration, ConfigurationSpace, EqualsCondition, Float, InCondition, Integer, Constant
from smac import HyperparameterOptimizationFacade, MultiFidelityFacade, Scenario
from smac.intensifier.hyperband import Hyperband


# Combination of pos/neg 
def join_df_metNames(df, grouper="peakID", include_mass=False):
    """
    Sets common index for combination of positively and negatively charged dataframes along their metabolite Names
    """
    mass_name = "ref_mass" if "ref_mass" in df.columns else "mass"
    data_cols = [col for col in df.columns[df.dtypes == "float"] if col not in [mass_name, "kegg", "dmz"]]
    cols = ["metNames"] + [f"MS{i+1}" for i in range(len(data_cols))]
    if include_mass:
        cols = cols + mass_name
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


# Helpers
def dict_permutations(dictionary:dict) -> List[dict]:
  """
  Combine all value combinations in a dictionary into a list of dictionaries.
  """
  keys, values = zip(*dictionary.items())
  return [dict(zip(keys, v)) for v in itertools.product(*values)]

'''
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
    """
    #Perform Grid parameter search for a 
    """
    grids = {}
    for i, y in enumerate(ys.transpose()):
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
'''

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
    def __init__(self, X, ys, cv:int, configuration_space:ConfigurationSpace, classifier, n_trials):
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
            splitter = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=seed)
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


def tune_classifier(X, y, classifier, cv, configuration_space, n_trials, name, algorithm_name, outdir, verbosity):
    """
    Perform hyperparameter tuning on an Sklearn classifier.
    """
    classifier = SKL_Classifier(X, y, cv=cv, configuration_space=configuration_space, classifier=classifier, n_trials=n_trials)

    scenario = Scenario( classifier.configuration_space, deterministic=True, 
                         n_workers=1, n_trials=n_trials,
                         walltime_limit=np.inf, cputime_limit=np.inf, trial_memory_limit=None,
                         output_directory=outdir )

    facade = HyperparameterOptimizationFacade(scenario, classifier.train, overwrite=True, logging_level=30-verbosity*10)

    mlflow.set_tracking_uri(Path(os.path.join(outdir, "mlruns")))
    mlflow.set_experiment(f"{name}_{algorithm_name}")
    with mlflow.start_run(run_name=f"{name}_{algorithm_name}"):
        mlflow.set_tag("test_identifier", "parent")
        incumbent = facade.optimize()

    return incumbent


def extract_metrics(true_labels, prediction, run_label=None, cv_i=None,
                    metrics_df:pd.DataFrame=pd.DataFrame(columns=["Run", "Cross-Validation run", "Accuracy", "AUC", "TPR", "FPR", "Threshold", "Conf_Mat"])):
    """
    Extract meaningful metrics to score a model according to its label prediction in comparison to true labels.
    """
    fpr, tpr, threshold = roc_curve(true_labels,  prediction)
    auc = roc_auc_score(true_labels,  prediction)
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
                                        inner_fold:int=3, verbosity=0):
    """
    Cross-validate a model against the given hyperparameters for all organisms
    """
    metrics_df = pd.DataFrame(columns=["Organism", "Cross-Validation run", "Accuracy", "AUC", "TPR", "FPR", "Threshold", "Conf_Mat"])
    organism_metrics_df = pd.DataFrame(columns=["Organism", "Accuracy", "AUC", "TPR", "FPR", "Threshold", "Conf_Mat"])
    overall_metrics_df = pd.DataFrame(columns=["Accuracy", "AUC", "TPR", "FPR", "Threshold", "Conf_Mat"])

    all_predictions = np.ndarray((0))
    ys_deshuffeld = np.ndarray((0))

    # Iterate over all organisms for binary distinction
    for i, y in enumerate(tqdm(ys.columns)):
        y = ys[y]
        predictions = np.ndarray((0))
        y_deshuffled = np.ndarray((0))

        # Outer Loop
        for cv_i, (train_index, val_index) in enumerate(fold.split(X, y)):
            training_data = X.iloc[train_index]
            training_labels = y.iloc[train_index]
            validation_data = X.iloc[val_index]
            validation_labels = y.iloc[val_index]

            # Hyperparameter Tuning with inner CV loop
            incumbent = tune_classifier(training_data, training_labels, classifier, inner_fold, configuration_space, n_trials,
                                        name, algorithm_name, outdir, verbosity)
            
            # Model definition and fitting
            best_hp = extract_best_hyperparameters_from_incumbent(incumbent=incumbent, configuration_space=configuration_space)
            best_hp = individual_layers_to_tuple(best_hp)
            model = classifier(**best_hp)		# Ensures model resetting for each cross-validation

            model.fit(np.array(training_data), np.array(training_labels))

            # Prediction and scoring
            prediction = model.predict(np.array(validation_data))
            
            metrics_df = extract_metrics(validation_labels, prediction, labels[i], cv_i+1, metrics_df)
			
            if verbosity != 0:
                if hasattr(model, "evaluate"):
                    model.evaluate(validation_data,  validation_labels, verbose=verbosity)
                if hasattr(model, "score"):
                    print(f"Mean accuracy: {model.score(validation_data,  validation_labels)}")

            predictions = np.append(predictions, prediction)
            y_deshuffled = np.append(y_deshuffled, validation_labels)

        organism_metrics_df = extract_metrics(y_deshuffled, predictions, labels[i], metrics_df=organism_metrics_df)
        all_predictions = np.append(all_predictions, predictions)
        ys_deshuffeld = np.append(ys_deshuffeld, y_deshuffled)

    overall_metrics_df = extract_metrics(ys_deshuffeld, all_predictions, metrics_df=overall_metrics_df)

    # Saving of results
    metrics_df.to_csv(os.path.join(outdir, f"{algorithm_name}_metrics.tsv"), sep="\t")
    organism_metrics_df.to_csv(os.path.join(outdir, f"{algorithm_name}_organism_metrics.tsv"), sep="\t")
    overall_metrics_df.to_csv(os.path.join(outdir, f"{algorithm_name}_overall_metrics.tsv"), sep="\t")

    return (metrics_df, organism_metrics_df, overall_metrics_df)


def cross_validate_train_model_sklearn( X, ys, labels, classifier, configuration_space, n_trials,
                                        name, algorithm_name, outdir, fold:Union[KFold, StratifiedKFold]=KFold(),
                                        verbosity=0 ):
    """
    Cross-validate a model against the given hyperparameters for all organisms
    """
    # Iterate over all organisms for binary distinction
    for i, org in enumerate(tqdm(ys.columns)):
        y = ys[org]

        incumbent = tune_classifier(X, y, classifier, fold, configuration_space, n_trials,
                                    name, algorithm_name, outdir, verbosity)
        
        # Model definition and fitting
        best_hp = extract_best_hyperparameters_from_incumbent(incumbent=incumbent, configuration_space=configuration_space)

        best_hp = individual_layers_to_tuple(best_hp)
        model = classifier(**best_hp)		# Ensures model resetting for each cross-validation

        model.fit(np.array(X), np.array(y))
        with open(os.path.join(outdir, f'model_{labels[i]}.pkl'),'wb') as f:
            pickle.dump(model ,f)


# Keras
def build_classification_model(config:Configuration, classes:int=1):
    backend.clear_session()
    gc.collect()

    # Model definition
    model = keras.Sequential(name="MS_community_classifier")
    model.add( keras.layers.Dropout( config["dropout_in"] ) )
    model.add( keras.layers.BatchNormalization() )
    for i in range( config["n_layers"] ):
        activation = config[f"activation_{i}"]
        if activation == "leakyrelu":
            activation = layers.LeakyReLU()
        model.add( layers.Dense( units=config[f"n_neurons_{i}"], activation=activation)  )
        if config[f"dropout_{i}"]:
            model.add( keras.layers.Dropout(0.25, noise_shape=None, seed=None) )
        model.add( keras.layers.BatchNormalization() )

    model.add( layers.Dense(classes, activation=activations.sigmoid) )

    if classes == 1:
        loss_function = keras.losses.BinaryCrossentropy()
    else:
        loss_function = keras.losses.CategoricalCrossentropy()

    if config["solver"] == "nadam":
        optimizer = keras.optimizers.Nadam( learning_rate=config["learning_rate"] )
    
    model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])

    return model

class Keras_Classifier:
    def __init__(self, X, ys, cv, configuration_space:ConfigurationSpace, model_builder, model_args, n_trials):
        self.configuration_space = configuration_space
        self.model_builder = model_builder
        self.model_args = model_args
        self.fold = KFold(n_splits=cv)
        self.X = X
        self.ys = ys
        self.progress_bar = tqdm(total=n_trials)

    def train(self, config: Configuration, seed: int = 0, budget: int = 25) -> np.float64:
        hls = individual_layers_to_tuple(config)["hidden_layer_sizes"]
        self.progress_bar.set_postfix_str(f'Connection size: {np.sum([hls[i] * hls[i+1] for i in range(len(hls) - 1)])}')

        keras.utils.set_random_seed(seed)
        losses = []
        for train_index, val_index in self.fold.split(self.X, self.ys):
            model = self.model_builder(config=config, **self.model_args)
            model.fit(self.X.iloc[train_index], self.ys.iloc[train_index], epochs=int(budget), verbose=0)
            val_loss, val_acc = model.evaluate(self.X.iloc[val_index],  self.ys.iloc[val_index], verbose=0)
            losses.append(val_loss)
        model.summary()
        self.progress_bar.update(1)       

        return np.mean(losses)


# Evaluation
def nested_cross_validate_model_keras(X, ys, labels, configuration_space, n_trials=100, classes=1, fold:Union[KFold, StratifiedKFold]=KFold(),
                                      patience:int=100, epochs:int=1000, outdir=".", verbosity=0):
    """
    Perform nested cross-validation with hyperparameter search on the given configuration space and subsequent evaluation
    """
    metrics_df = pd.DataFrame(columns=["Organism", "Cross-Validation run", "Accuracy", "AUC", "TPR", "FPR", "Threshold", "Conf_Mat"])
    organism_metrics_df = pd.DataFrame(columns=["Organism", "Accuracy", "AUC", "TPR", "FPR", "Threshold", "Conf_Mat"])
    overall_metrics_df = pd.DataFrame(columns=["Accuracy", "AUC", "TPR", "FPR", "Threshold", "Conf_Mat"])

    all_predictions = np.ndarray((0))

    for i, y in enumerate(tqdm(ys.columns)):
        y = ys[y]
        predictions = np.ndarray((0))

        for cv_i, (train_index, val_index) in enumerate(tqdm(fold.split(X, y))):
            training_data = X.iloc[train_index]
            training_labels = y.iloc[train_index]
            validation_data = X.iloc[val_index]
            validation_labels = y.iloc[val_index]

            classifier = Keras_Classifier( training_data, training_labels, cv=3, configuration_space=configuration_space,
                                        model_builder=build_classification_model, model_args={"classes": 1}, n_trials=n_trials )

            scenario = Scenario( classifier.configuration_space, n_trials=n_trials,
                                deterministic=True,
                                min_budget=5, max_budget=100,
                                n_workers=1, output_directory=outdir,
                                walltime_limit=np.inf, cputime_limit=np.inf, trial_memory_limit=None    # Max RAM in Bytes (not MB) 3600 = 1h
                                )

            initial_design = MultiFidelityFacade.get_initial_design( scenario, n_configs=100 )
            intensifier = Hyperband( scenario, incumbent_selection="highest_budget" )
            facade = MultiFidelityFacade( scenario, classifier.train, 
                                        initial_design=initial_design, intensifier=intensifier,
                                        overwrite=True, logging_level=20 )
            
            
            incumbent = facade.optimize()
            
            best_hp = extract_best_hyperparameters_from_incumbent(incumbent=incumbent, configuration_space=configuration_space)
            model = build_classification_model(best_hp, classes)		# Ensures model resetting for each cross-validation

            callback = keras.callbacks.EarlyStopping(monitor='loss', patience=patience)
            model.fit(training_data, training_labels, epochs=epochs, verbose=0, callbacks=[callback]) # type: ignore

            prediction = np.where( model.predict(validation_data) > 0.5, 1.0, 0.0)

            metrics_df = extract_metrics(validation_labels, prediction, labels[i], cv_i+1, metrics_df)
			
            if verbosity != 0:
                model.evaluate(validation_data,  validation_labels, verbose=verbosity)

            predictions = np.append(predictions, prediction)
            keras.backend.clear_session()
            
        organism_metrics_df = extract_metrics(y, predictions, labels[i], metrics_df=organism_metrics_df)
        all_predictions = np.append(all_predictions, predictions)

    overall_metrics_df = extract_metrics(ys.to_numpy().flatten(), all_predictions, metrics_df=overall_metrics_df)
    return (metrics_df, organism_metrics_df, overall_metrics_df)



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


def plot_metrics_df(metrics_df, organism_metrics_df, overall_metrics_df, algorithm_name, outdir, show=True):
    """
    Plot the extracted metrics
    """
    ax = sns.heatmap(metrics_df.pivot(index="Organism", columns="Cross-Validation run", values="Accuracy"),
                    vmin=0, vmax=1.0, annot=True, cmap=sns.diverging_palette(328.87,  221.63, center="light", as_cmap=True))
    fig = ax.get_figure()
    fig.savefig(os.path.join(outdir, f"{algorithm_name}_heatmap_accuracies.svg"), bbox_inches="tight")
    plt.show()

    ax = sns.lineplot( overall_metrics_df.explode(["TPR", "FPR"]), x="TPR", y="FPR", hue="AUC")
    ax = sns.lineplot( organism_metrics_df.explode(["TPR", "FPR"]), x="TPR", y="FPR", hue="Organism", alpha=0.5, ax=ax)
    ax.set_title("AUC")
    leg = ax.axes.get_legend()
    leg.set_title("Organism (AUC)")
    for t, l in zip(leg.texts, [f'{row["Organism"]} (AUC={str(row["AUC"])})' for i, row in organism_metrics_df.iterrows()]+ [f"Overall (AUC={overall_metrics_df.loc[0, 'AUC']})"]):
        t.set_text(l)
    fig = ax.get_figure()
    fig.savefig(os.path.join(outdir, f"{algorithm_name}_AUC.svg"), bbox_inches="tight")
    if show:
        plt.show()