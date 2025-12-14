% 1. Load the Dataset
% We use readtable/readmatrix to handle headers safely
try
    data = readmatrix('koa_features.csv');
catch
    % Fallback if using older MATLAB
    data = csvread('koa_features.csv', 1, 0); 
end

% 2. Prepare Data
% Features (First 16 columns)
X = data(:, 1:end-1); 
% Labels (Last column) - Must be categorical for classification
Y = categorical(data(:, end)); 

% 3. Define the Network Architecture
% Since we have 1D data (features), we use Fully Connected layers, not 2D Conv
numFeatures = size(X, 2); % Should be 16
numClasses = numel(categories(Y)); % Should be 5 (Grades 0-4)

layers = [
    featureInputLayer(numFeatures, 'Normalization', 'zscore', 'Name', 'input')
    
    fullyConnectedLayer(64, 'Name', 'fc1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')
    
    fullyConnectedLayer(32, 'Name', 'fc2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
    
    fullyConnectedLayer(numClasses, 'Name', 'fc_out')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'classoutput')
];

% 4. Training Options
options = trainingOptions('adam', ...
    'MaxEpochs', 50, ...
    'MiniBatchSize', 32, ...
    'InitialLearnRate', 0.001, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress', ...
    'Verbose', false);

% 5. Train the Network
fprintf('Training Neural Network...\n');
net = trainNetwork(X, Y, layers, options);

% 6. Evaluate
YPred = classify(net, X);
accuracy = sum(YPred == Y) / numel(Y);

% 7. Display Results
fprintf('---------------------------------\n');
fprintf('Final Accuracy: %.2f%%\n', accuracy * 100);
fprintf('---------------------------------\n');

% Show Confusion Matrix
figure;
confusionchart(Y, YPred);
title('Neural Network Classification Results');