function [feature_vector] = haralick(I)

I = im2gray(I);

% Define GLCM properties
offsets = [0 1; -1 1; -1 0; -1 -1];
numlevels = 8;
symmetric = true;
glcm = graycomatrix(I, 'Offset', offsets, 'NumLevels', numlevels, 'Symmetric', symmetric);

% Normalize GLCM
glcm_norm = glcm / sum(glcm(:));

% Rescale normalized GLCM to integer values
glcm_rescaled = round(glcm_norm * 255);

% Calculate Haralick texture features
stats = graycoprops(glcm_rescaled, {'contrast', 'correlation', 'energy', 'homogeneity'});

% Display results
disp(stats.Contrast);
disp(stats.Correlation);
disp(stats.Energy);
disp(stats.Homogeneity);

feature_vector=[stats.Contrast stats.Correlation stats.Energy stats.Homogeneity]