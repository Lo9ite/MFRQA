% TRAIN_DEMO Example script to train MFRQA SVR model from a CSV file.
%
% CSV format (with headers):
%   enhanced_path,lowlight_path,score
%
% Example:
%   data = readtable('train_pairs.csv');
%   [model, feats] = train_MFRQA_SVR_fn(data.enhanced_path, data.lowlight_path, ...
%       data.score, 'C', 8, 'Gamma', 0.5, 'Epsilon', 0.1, 'ModelPath', 'model_mfrqa_svr.mat');

clear;
clc;

csvPath = 'train_pairs.csv';
if ~isfile(csvPath)
    error('Cannot find %s. Please create it first.', csvPath);
end

data = readtable(csvPath, 'TextType', 'string');

requiredCols = {'enhanced_path', 'lowlight_path', 'score'};
for i = 1:numel(requiredCols)
    if ~ismember(requiredCols{i}, data.Properties.VariableNames)
        error('Missing required column: %s', requiredCols{i});
    end
end

if any(isnan(data.score))
    error(['Column "score" contains NaN values. ' ...
           'Please replace NaN with valid MOS/DMOS labels before training.']);
end

[model, FeatureTrain] = train_MFRQA_SVR_fn(cellstr(data.enhanced_path), ...
                                        cellstr(data.lowlight_path), ...
                                        data.score, ...
                                        'C', 8, ...
                                        'Gamma', 0.5, ...
                                        'Epsilon', 0.1, ...
                                        'ModelPath', 'model_mfrqa_svr.mat');

fprintf('Feature matrix size: %d x %d\n', size(FeatureTrain, 1), size(FeatureTrain, 2));