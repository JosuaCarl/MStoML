
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