# imports
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split, KFold
import tensorflow as tf
import keras
from keras import layers, activations
from ConfigSpace import Categorical, Configuration, ConfigurationSpace, EqualsCondition, Float, InCondition, Integer
from smac import MultiFidelityFacade, HyperparameterOptimizationFacade
from smac import Scenario
from smac.intensifier.hyperband import Hyperband

sys.path.append( '..' )
from helpers import *


def build_classification_model(config:Configuration, multiclass:bool=False):
    # Model definition
    model = keras.Sequential(name="MS_community_classifier")
    model.add(keras.layers.Dropout( config.get("dropout_in") ))
    model.add(keras.layers.BatchNormalization())
    for i in range( config.get("n_layers") ):
        activation = config.get(f"activation_{i}")
        if activation == "leakyrelu":
            activation = layers.LeakyReLU()
        model.add( layers.Dense( units=config.get(f"n_neurons_{i}"), activation=activation ) )
        if config.get(f"dropout_{i}"):
            model.add(keras.layers.Dropout(0.5, noise_shape=None, seed=None))
        model.add(keras.layers.BatchNormalization())

    if multiclass:
        model.add(layers.Dense(8,  activation=activations.sigmoid))     # Interpretation layer
        loss_function = keras.losses.CategoricalCrossentropy()
    else:
        model.add(layers.Dense(1,  activation=activations.sigmoid))     # Interpretation layer
        loss_function = keras.losses.BinaryCrossentropy()

    if config.get("solver") == "nadam":
        optimizer = keras.optimizers.Nadam( learning_rate=config.get("learning_rate") )
    
    model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])

    return model


class classifier:
    def __init__(self, X, ys, test_size:float, configuration_space:ConfigurationSpace, model_builder, model_args):
        self.configuration_space = configuration_space
        self.model_builder = model_builder
        self.model_args = model_args
        self.training_data, self.test_data, self.training_labels, self.test_labels = train_test_split(X, ys, test_size=test_size)

    @property
    def configspace(self) -> ConfigurationSpace:
        return self.configuration_space

    def train(self, config: Configuration, seed: int = 0, budget: int = 25) -> float:
            keras.utils.set_random_seed(seed)
            model = self.model_builder(config=config, **self.model_args)
    
            callback = keras.callbacks.EarlyStopping(monitor='loss', patience=100)	# Model will stop if no improvement
            model.fit(self.training_data, self.training_labels, epochs=int(budget), verbose=0, callbacks=[callback])

            val_loss, val_acc = model.evaluate(self.test_data,  self.test_labels, verbose=0)

            return val_loss
    

def cross_validate_model(X, y, model, fold=KFold(), patience=100, epochs=1000, verbosity=0):
	"""
	Cross-validate a model against the given hyperparameters for all organisms
	"""
	metrics_df = pd.DataFrame(columns=["Organism", "Cross-Validation run", "Accuracy", "AUC", "TPR", "FPR", "Threshold", "Conf_Mat"])

	for i, y in enumerate(tqdm(ys.columns)):
		y = ys[y]
		for cv_i, (train_index, val_index) in enumerate(fold.split(X, y)):
			model_acc = model		# Ensures model resetting for each cross-validation
			training_data = X.iloc[train_index]
			training_labels = y.iloc[train_index]
			validation_data = X.iloc[val_index]
			validation_labels = y.iloc[val_index]
			
			callback = keras.callbacks.EarlyStopping(monitor='loss', patience=patience)
			model_acc.fit(training_data, training_labels, epochs=epochs, verbose=0, callbacks=[callback])

			prediction = model_acc.predict(validation_data)
			metrics_df = extract_metrics(validation_labels, prediction, strains.iloc[i].item(), cv_i+1, metrics_df)
			
			if verbosity != 0:
				model_acc.evaluate(validation_data,  validation_labels, verbose=verbosity)

	return metrics_df