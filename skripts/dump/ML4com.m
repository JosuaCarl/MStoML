function ML4com

% Load or define your matrices
load('strains.mat')
load('comm8.mat')
load('metabolome_COM8.mat')

M = metData;

% randomize 
n_ix = randperm(68);

% C = C(n_ix,:);

% Number of folds for cross-validation
numFolds = 6;

% Initialize Random Forest model
nTrees = 100; % Number of trees in the Random Forest

% Initialize storage for cross-validation accuracy
cvAccuracies = zeros(size(C, 2), numFolds);

indices = crossvalind('Kfold', C(:, 1), numFolds);

% Perform cross-validation for each strain separately
for i = 1:size(C, 2)
    for j = 1:numFolds
        % Define training and validation data
        test = (indices == j);
        train = ~test;
        
        % Train the Random Forest model
        rfModel = TreeBagger(nTrees, M(:, train)', C(train, i), 'Method', 'classification',...
             'PredictorSelection','curvature','OOBPredictorImportance','on');
         
        %%   predictors
        model(i).imp(:,j) = rfModel.OOBPermutedPredictorDeltaError;
        
        % Predict on validation data
        predictions = predict(rfModel, M(:, test)');
        
        % Convert predictions to binary
        predictions = cellfun(@str2double, predictions);
        predictions(predictions >= 0.5) = 1;
        predictions(predictions < 0.5) = 0;
        
        % Calculate and store accuracy
        accuracy = sum(predictions == C(test, i)) / length(predictions);
        cvAccuracies(i, j) = accuracy;
        
    end
end

% Display cross-validation results
for i = 1:size(C, 2)
    fprintf([ strains{i} ' - Cross-validation Accuracies: ']);
    fprintf('%.2f%% ', cvAccuracies(i, :) * 100);
    fprintf('\nAverage Accuracy: %.2f%%\n', mean(cvAccuracies(i, :)) * 100);
end