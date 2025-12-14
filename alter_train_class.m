clc; clear;

% Load CSV
data = readtable('koa_features.csv');

% Remove filename column if it is non-numeric
if ~isnumeric(data{1,end})
    data(:, end) = [];
end

% Extract features and labels
X = data{:, 1:end-1};   % all columns except last
Y = data{:, end};       % last column is label

% Convert labels to categorical
Y = categorical(Y);

% MANUAL Train/Test Split (80% train, 20% test)
N = size(X,1);
idx = randperm(N);

trainCount = round(0.8 * N);

trainIdx = idx(1:trainCount);
testIdx  = idx(trainCount+1:end);

Xtrain = X(trainIdx, :);
Ytrain = Y(trainIdx);

Xtest  = X(testIdx, :);
Ytest  = Y(testIdx);

% Train a decision tree (fitctree works without toolbox)
model = fitctree(Xtrain, Ytrain);

% Predict
Ypred = predict(model, Xtest);

% Accuracy
accuracy = mean(Ypred == Ytest);
fprintf('Test Accuracy: %.2f%%\n', accuracy * 100);

% Confusion Matrix
figure;
confusionchart(Ytest, Ypred);
title('Confusion Matrix - KOA Decision Tree Classifier');
