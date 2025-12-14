clc;
close all;

%% 1. SETTINGS
% Change this to your actual image file name
imageFileName = 'grade_3.png'; % <--- PUT YOUR IMAGE PATH HERE

%% 2. LOAD MODEL
% Check if the trained model 'net' is in the workspace.
% If not, try to load it or warn the user.
if exist('net', 'var')
    fprintf('Model "net" found in workspace.\n');
elseif exist('my_best_koa_model.mat', 'file')
    load('my_best_koa_model.mat');
    fprintf('Loaded model from file.\n');
else
    error('Error: Trained model "net" not found! Please run "train_neural_network.m" first.');
end

%% 3. PROCESS THE IMAGE
fprintf('Processing image: %s ...\n', imageFileName);

try
    % Read image
    I = imread(imageFileName);
    
    % Show the image
    figure;
    imshow(I);
    title('Testing Image');
    
    % Convert to Grayscale if needed
    if size(I, 3) == 3
        I = rgb2gray(I);
    end
    
    % Resize? (Optional: Your training script didn't seem to resize before Haralick, 
    % but if your images are huge, you might want to. 
    % However, Haralick is texture-based, so size matters less than in CNNs).
    
    % EXTRACT FEATURES
    % We use the EXACT same function used for training
    features = haralick(I);
    
    fprintf('Features Extracted: [Contrast: %.2f, Correlation: %.2f, ...]\n', features(1), features(2));
    
    %% 4. PREDICT
    % The model expects a table or matrix. Since we have a single vector (1x16),
    % we pass it directly.
    
    prediction = classify(net, features);
    
    %% 5. DISPLAY RESULT
    fprintf('\n======================================\n');
    fprintf(' FINAL DIAGNOSIS \n');
    fprintf('======================================\n');
    fprintf(' Predicted Grade: %s\n', string(prediction));
    fprintf('======================================\n');
    
    title(['AI Prediction: Grade ' char(string(prediction))]);

catch ME
    fprintf('Error processing image: %s\n', ME.message);
    fprintf('Make sure the filename is correct and the image exists.\n');
end