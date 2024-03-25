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
import keras_tuner
from ConfigSpace import Categorical, Configuration, ConfigurationSpace, EqualsCondition, Float, InCondition, Integer, Constant
from smac import MultiFidelityFacade, HyperparameterOptimizationFacade
from smac import Scenario
from smac.intensifier.hyperband import Hyperband

sys.path.append( '..' )
from helpers import *
from ML4com import *

# Models
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

class Classifier:
    def __init__(self, X, ys, test_size:float, configuration_space:ConfigurationSpace, model_builder, model_args):
        self.configuration_space = configuration_space
        self.model_builder = model_builder
        self.model_args = model_args
        self.training_data, self.test_data, self.training_labels, self.test_labels = train_test_split(X, ys, test_size=test_size)

    def train(self, config: Configuration, seed: int = 0, budget: int = 25) -> np.float64:
        model = self.model_builder(config=config, **self.model_args)
        losses = []
        for y in self.training_labels.columns:
            keras.utils.set_random_seed(seed)
    
            y_train = self.training_labels[y]
            y_test= self.test_labels[y]
            
            callback = keras.callbacks.EarlyStopping(monitor='loss', patience=100)
            model.fit(self.training_data, y_train, epochs=int(budget), verbose=0, callbacks=[callback])

            val_loss, val_acc = model.evaluate(self.test_data,  y_test, verbose=0)
            losses.append(val_loss)
            keras.backend.clear_session()

        return np.mean(losses)


# Evaluation
def extract_metrics(true_labels, prediction, run_label, cv_i,
                    metrics_df:pd.DataFrame=pd.DataFrame(columns=["Run", "Cross-Validation run", "Accuracy", "AUC", "TPR", "FPR", "Threshold", "Conf_Mat"])):
    fpr, tpr, threshold = roc_curve(true_labels,  prediction)
    auc = roc_auc_score(true_labels,  prediction)

    prediction_labels = [0.0 if pred[0] < 0.5 else 1.0 for pred in prediction]
    conf_mat = confusion_matrix(true_labels,  prediction_labels)
    accuracy = accuracy_score(true_labels,  prediction_labels)

    metrics_df.loc[len(metrics_df)] = [run_label, cv_i, accuracy, auc, tpr, fpr, threshold, conf_mat]

    return metrics_df

def cross_validate_model(X, ys, labels, config, classes=1, fold:Union[KFold, StratifiedKFold]=KFold(), patience:int=100, epochs:int=1000, verbosity=0):
    """
    Cross-validate a model against the given hyperparameters for all organisms
    """
    metrics_df = pd.DataFrame(columns=["Organism", "Cross-Validation run", "Accuracy", "AUC", "TPR", "FPR", "Threshold", "Conf_Mat"])

    for i, y in enumerate(tqdm(ys.columns)):
        y = ys[y]
        for cv_i, (train_index, val_index) in enumerate(fold.split(X, y)):
            model = build_classification_model(config, classes)		# Ensures model resetting for each cross-validation
            training_data = X.iloc[train_index]
            training_labels = y.iloc[train_index]
            validation_data = X.iloc[val_index]
            validation_labels = y.iloc[val_index]

            callback = keras.callbacks.EarlyStopping(monitor='loss', patience=patience)
            model.fit(training_data, training_labels, epochs=epochs, verbose=0, callbacks=[callback]) # type: ignore

            prediction = model.predict(validation_data)
            metrics_df = extract_metrics(validation_labels, prediction, labels[i], cv_i+1, metrics_df)
			
            if verbosity != 0:
                model.evaluate(validation_data,  validation_labels, verbose=verbosity) # type: ignore

            keras.backend.clear_session()
    return metrics_df