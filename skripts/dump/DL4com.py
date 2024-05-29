# imports
import sys
import gc
import os

import sklearn
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['MALLOC_TRIM_THRESHOLD_'] = '0'

from typing import Union
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
import tensorflow as tf
import keras
from keras import layers, activations, backend
from ConfigSpace import Categorical, Configuration, ConfigurationSpace, EqualsCondition, Float, InCondition, Integer, Constant
from smac import MultiFidelityFacade, HyperparameterOptimizationFacade
from smac import Scenario
from smac.intensifier.hyperband import Hyperband
import smac

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append( os.path.join( dir_path, '../..' ))
from skripts.helpers.mailing import *
from ML4com import *

# Models
'''
def build_classification_model_kt(hyperparameters, classes:int=8):
    backend.clear_session()
    gc.collect()
    model = keras.Sequential(name="MS_community_classifier")
    
    model.add(keras.layers.Dropout(hyperparameters.Float("dropout_in", min_value=0.2, max_value=0.7, sampling="linear"),
                                   noise_shape=None, seed=None))
    model.add(keras.layers.BatchNormalization())
    
    # Middle layers
    for i in range(hyperparameters.Int("num_layers", 1, 5)):
        activation = hyperparameters.Choice(f"activation_{i}", ["relu", "leakyrelu"])
        if activation == "leakyrelu":
            activation = layers.LeakyReLU()
        model.add(
            layers.Dense(
                units=hyperparameters.Int(f"units_{i}", min_value=10, max_value=1010, step=500),
                activation=activation,
            )
        )
        if hyperparameters.Boolean(f"dropout_{i}"):
            model.add(keras.layers.Dropout(0.5, noise_shape=None, seed=None))
        model.add(keras.layers.BatchNormalization())

    model.add(layers.Dense(classes,  activation=activations.sigmoid))
    if classes == 1:
        loss_function = keras.losses.BinaryCrossentropy()
    else:
        loss_function = keras.losses.CategoricalCrossentropy()
    
    loss_function = keras.losses.CategoricalCrossentropy()
    optimizer = keras.optimizers.Nadam(learning_rate=1e-3)

    model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])
    return model
'''

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
            model.add(keras.layers.Dropout(0.5, noise_shape=None, seed=None))
        model.add(keras.layers.BatchNormalization())

    model.add(layers.Dense(classes,  activation=activations.sigmoid))
    if classes == 1:
        loss_function = keras.losses.BinaryCrossentropy()
    else:
        loss_function = keras.losses.CategoricalCrossentropy()

    if config["solver"] == "nadam":
        optimizer = keras.optimizers.Nadam( learning_rate=config["learning_rate"] )
    
    model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])

    return model

class Keras_Classifier:
    def __init__(self, X, ys, cv, configuration_space:ConfigurationSpace, model_builder, model_args):
        self.configuration_space = configuration_space
        self.model_builder = model_builder
        self.model_args = model_args
        self.fold = KFold(n_splits=cv)
        self.X = X
        self.ys = ys

    def train(self, config: Configuration, seed: int = 0, budget: int = 25) -> np.float64:
        model = self.model_builder(config=config, **self.model_args)
        losses = []
        for train_index, val_index in self.fold.split(self.X, self.ys):
            training_data = self.X.iloc[train_index]
            training_labels = self.ys.iloc[train_index]
            validation_data = self.X.iloc[val_index]
            validation_labels = self.ys.iloc[val_index]

            for y in training_labels.columns:
                keras.utils.set_random_seed(seed)
        
                y_train = training_labels[y]
                y_test= validation_labels[y]
                
                callback = keras.callbacks.EarlyStopping(monitor='loss', patience=100)
                model.fit(training_data, y_train, epochs=int(budget), verbose=0, callbacks=[callback])

                val_loss, val_acc = model.evaluate(validation_data,  y_test, verbose=0)
                losses.append(val_loss)
                keras.backend.clear_session()

        return np.mean(losses)


# Evaluation
def nested_cross_validate_model_keras(X, ys, labels, configuration_space, classes=1, fold:Union[KFold, StratifiedKFold]=KFold(),
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
        for cv_i, (train_index, val_index) in enumerate(fold.split(X, y)):
            training_data = X.iloc[train_index]
            training_labels = y.iloc[train_index]
            validation_data = X.iloc[val_index]
            validation_labels = y.iloc[val_index]

            classifier = Keras_Classifier(training_data, ys, cv=3 , configuration_space=configuration_space,
                                    model_builder=build_classification_model, model_args={"classes": 1})

            scenario = Scenario( classifier.configuration_space, n_trials=1000,
                                deterministic=True,
                                min_budget=5, max_budget=1000,
                                n_workers=1, output_directory=outdir,
                                walltime_limit=12*60*60, cputime_limit=np.inf, trial_memory_limit=int(6e10)    # Max RAM in Bytes (not MB) 3600 = 1h
                                )

            initial_design = MultiFidelityFacade.get_initial_design(scenario, n_configs=100)
            intensifier = Hyperband(scenario, incumbent_selection="highest_budget")
            facade = MultiFidelityFacade( scenario, classifier.train, 
                                        initial_design=initial_design, intensifier=intensifier,
                                        overwrite=True, logging_level=20
                                        )

            incumbent = facade.optimize()

            if isinstance(incumbent, list):
                best_hp = incumbent[0]
            else: 
                best_hp = incumbent

            model = build_classification_model(best_hp, classes)		# Ensures model resetting for each cross-validation

            callback = keras.callbacks.EarlyStopping(monitor='loss', patience=patience)
            model.fit(training_data, training_labels, epochs=epochs, verbose=0, callbacks=[callback]) # type: ignore

            prediction = model.predict(validation_data)
            metrics_df = extract_metrics(validation_labels, prediction, labels[i], cv_i+1, metrics_df)
			
            if verbosity != 0:
                model.evaluate(validation_data,  validation_labels, verbose=verbosity) # type: ignore

            predictions = np.append(predictions, prediction)
            keras.backend.clear_session()
            
        organism_metrics_df = extract_metrics(y, predictions, labels[i], metrics_df=organism_metrics_df)
        all_predictions = np.append(all_predictions, predictions)

    overall_metrics_df = extract_metrics(ys.to_numpy().flatten(), all_predictions, metrics_df=overall_metrics_df)
    return (metrics_df, organism_metrics_df, overall_metrics_df)