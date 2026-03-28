% SCORE_LIEQ Run MFRQA scoring on paired images from the LIEQ dataset.
%
% LIEQ expected structure:
%   H:\lijin\NLIEE-main\LIEQ
%     dataset
%       LowLight
%         001.png ... 100.png
%       Enhancement
%         001_bpdhe.png, 001_CRM.png, ..., 100_SRIE.png
%
% Output:
%   - Prints progress and scores in command window.
%   - Saves full results to: score_LIEQ.csv (in current folder).

clear;
clc;

% Change this path if needed.
datasetRoot = 'H:\lijin\NLIEE-main\LIEQ';

lowDir = fullfile(datasetRoot, 'dataset', 'LowLight');
enhDir = fullfile(datasetRoot, 'dataset', 'Enhancement');

if ~isfolder(lowDir)
    error('LowLight directory not found: %s', lowDir);
end
if ~isfolder(enhDir)
    error('Enhancement directory not found: %s', enhDir);
end

algorithms = {'bpdhe', 'CRM', 'dong', 'EFF', 'he', 'LIME', ...
              'multi_fusion', 'NPEA', 'PLE', 'SRIE'};

numImages = 100;
numAlgs = numel(algorithms);
numRows = numImages * numAlgs;

lowlight_name = strings(numRows, 1);
enhanced_name = strings(numRows, 1);
algorithm = strings(numRows, 1);
score = nan(numRows, 1);

row = 0;
for i = 1:numImages
    lowName = sprintf('%03d.png', i);
    lowPath = fullfile(lowDir, lowName);

    if ~isfile(lowPath)
        warning('Missing LowLight image: %s', lowPath);
        continue;
    end

    lowImg = imread(lowPath);

    for k = 1:numAlgs
        alg = algorithms{k};
        enhName = sprintf('%03d_%s.png', i, alg);
        enhPath = fullfile(enhDir, enhName);

        row = row + 1;
        lowlight_name(row) = string(lowName);
        enhanced_name(row) = string(enhName);
        algorithm(row) = string(alg);

        if ~isfile(enhPath)
            warning('Missing enhancement image: %s', enhPath);
            continue;
        end

        enhImg = imread(enhPath);

        % MFRQA_features comment convention: img1 = enhanced, img2 = low-light
        score(row) = MFRQA_SVR(enhImg, lowImg);

        fprintf('[%4d/%4d] %s -> %.6f\n', row, numRows, enhName, score(row));
    end
end

results = table(lowlight_name(1:row), enhanced_name(1:row), ...
                algorithm(1:row), score(1:row));
writetable(results, 'score_LIEQ.csv');
fprintf('Done. Results saved to score_LIEQ.csv\n');
