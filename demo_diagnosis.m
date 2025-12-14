clc;
close all;

fprintf('============================================\n');
fprintf('   KNEE OSTEOARTHRITIS DIAGNOSIS SYSTEM\n');
fprintf('============================================\n');

%% 1. Load Data (If missing)
if ~exist('XTest', 'var')
    fprintf('Status: Loading patient database...\n');
    try
        % Try loading the high-accuracy file first
        data = readmatrix('optimized_koa_features.csv'); 
    catch
        % Fallback to standard file
        data = readmatrix('koa_features.csv');
    end
    
    % Prepare the data (Same steps as training)
    X = data(:, 1:end-1); 
    Y = categorical(data(:, end)); 
    
    % Re-create the Test Set
    cv = cvpartition(Y, 'HoldOut', 0.3);
    XTest = X(cv.test, :);
    YTest = Y(cv.test, :);
end

%% 2. Load Model (If missing)
if ~exist('net', 'var')
    if exist('my_best_koa_model.mat', 'file')
        load('my_best_koa_model.mat');
    else
        fprintf('\nERROR: Model file not found!\n');
        fprintf('Please run "train_neural_network" first to create the model.\n');
        return;
    end
end

%% 3. Run the Demo
% Pick a random patient from the database
random_idx = randi(size(XTest, 1));
patient_data = XTest(random_idx, :);
actual_grade = string(YTest(random_idx));

fprintf('Patient ID:       #%d\n', random_idx);
fprintf('Input Features:   [Contrast: %.2f, Energy: %.2f ...]\n', patient_data(1), patient_data(3));
pause(0.5); 

% Ask AI to predict
fprintf('AI Analysis:      Processing...\n');
pause(1); 

prediction = classify(net, patient_data);
predicted_grade = string(prediction);

%% 4. Show Final Result
fprintf('\n----------------------------------\n');
fprintf('      DIAGNOSIS REPORT\n');
fprintf('----------------------------------\n');
fprintf(' Actual Condition: Grade %s\n', actual_grade);
fprintf(' AI Diagnosis:     Grade %s\n', predicted_grade);

if actual_grade == predicted_grade
    fprintf(' Result:           CORRECT DIAGNOSIS ✅\n');
else
    fprintf(' Result:           INCORRECT DIAGNOSIS ❌\n');
end
fprintf('----------------------------------\n');