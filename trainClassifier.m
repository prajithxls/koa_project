% 1. Load your existing data
data = readtable('koa_features.csv');

% 2. Rename the columns to match what 'trainClassifier' expects
% (It needs 'feature1' through 'feature16' and 'class')
varNames = {'feature1','feature2','feature3','feature4','feature5','feature6',...
            'feature7','feature8','feature9','feature10','feature11','feature12',...
            'feature13','feature14','feature15','feature16','class'};
data.Properties.VariableNames = varNames;

% 3. Now call the function with the data
[trainedModel, accuracy] = trainClassifier(data);

% 4. Display the result
fprintf('Decision Tree Validation Accuracy: %.2f%%\n', accuracy * 100);