clc;
clear all;
close all;

mainFolder = fullfile('test');

% Initialize feature matrix
ft = [];

for grade = 0:4
    folderPath = fullfile(mainFolder, num2str(grade));
    
    % Read all image types
    files = [dir(fullfile(folderPath, '*.jpg'));
             dir(fullfile(folderPath, '*.jpeg'));
             dir(fullfile(folderPath, '*.png'));
             dir(fullfile(folderPath, '*.bmp'))];

    fprintf("Grade %d folder path: %s\n", grade, folderPath);
    fprintf("Files found: %d\n", length(files));

    for i = 1:length(files)
        filePath = fullfile(folderPath, files(i).name);
        fprintf("    Processing: %s\n", files(i).name);

        try
            I = imread(filePath);

            if size(I,3) == 3
                I = rgb2gray(I);
            end

            h = haralick(I);

            ft = [ft; h grade];

        catch ME
            fprintf("ERROR processing %s: %s\n", files(i).name, ME.message);
        end
    end
end

writetable(array2table(ft), 'koa_features.csv');

fprintf("\n====================================\n");
fprintf("Completed! Total rows in CSV: %d\n", size(ft,1));
fprintf("====================================\n");
