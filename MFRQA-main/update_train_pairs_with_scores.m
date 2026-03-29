function T = update_train_pairs_with_scores(csvPath, modelPath, baseDir)
%UPDATE_TRAIN_PAIRS_WITH_SCORES Predict scores and append to train_pairs.csv.
%
%   T = update_train_pairs_with_scores()
%   T = update_train_pairs_with_scores(csvPath, modelPath, baseDir)
%
% Inputs (optional):
%   csvPath   - CSV file path (default: 'train_pairs.csv')
%   modelPath - trained model path (default: 'model_mfrqa_svr.mat')
%   baseDir   - base directory to resolve relative image paths. If empty,
%               first try the CSV folder, then current folder.
%
% Required CSV columns:
%   enhanced_path, lowlight_path
%
% Behavior:
%   - Reads image pairs from CSV
%   - Predicts each pair with MFRQA_SVR_with_model
%   - Inserts/overwrites column 'mfrqa_pred_score'
%   - Writes updated table back to the same CSV file

if nargin < 1 || isempty(csvPath)
    csvPath = 'train_pairs.csv';
end
if nargin < 2 || isempty(modelPath)
    modelPath = 'model_mfrqa_svr.mat';
end
if nargin < 3
    baseDir = '';
end

if ~isfile(csvPath)
    error('CSV file not found: %s', csvPath);
end

T = readtable(csvPath, 'TextType', 'string');
requiredCols = {'enhanced_path', 'lowlight_path'};
for i = 1:numel(requiredCols)
    if ~ismember(requiredCols{i}, T.Properties.VariableNames)
        error('Missing required column: %s', requiredCols{i});
    end
end

csvFolder = fileparts(csvPath);
numRows = height(T);
pred = nan(numRows, 1);

fprintf('Scoring %d pairs using model: %s\n', numRows, modelPath);
for i = 1:numRows
    enhPath = resolve_img_path(T.enhanced_path(i), baseDir, csvFolder);
    lowPath = resolve_img_path(T.lowlight_path(i), baseDir, csvFolder);

    if ~isfile(enhPath) || ~isfile(lowPath)
        warning('Skip row %d: missing file(s). enh=%s, low=%s', i, enhPath, lowPath);
        continue;
    end

    imgEnh = imread(enhPath);
    imgLow = imread(lowPath);
    pred(i) = MFRQA_SVR_with_model(imgEnh, imgLow, modelPath);

    if mod(i, 20) == 0 || i == numRows
        fprintf('  %d / %d\n', i, numRows);
    end
end

if ismember('mfrqa_pred_score', T.Properties.VariableNames)
    T.mfrqa_pred_score = pred;
else
    if ismember('score', T.Properties.VariableNames)
        scoreIdx = find(strcmp(T.Properties.VariableNames, 'score'), 1, 'first');
        T = addvars(T, pred, 'After', scoreIdx, 'NewVariableNames', 'mfrqa_pred_score');
    else
        T = addvars(T, pred, 'NewVariableNames', 'mfrqa_pred_score');
    end
end

writetable(T, csvPath);
fprintf('Done. Updated CSV written to: %s\n', csvPath);
end

function p = resolve_img_path(pathCell, baseDir, csvFolder)
pathStr = char(pathCell);
if is_absolute_path(pathStr) && isfile(pathStr)
    p = pathStr;
    return;
end

if ~isempty(baseDir)
    p1 = fullfile(baseDir, pathStr);
    if isfile(p1)
        p = p1;
        return;
    end
end

if ~isempty(csvFolder)
    p2 = fullfile(csvFolder, pathStr);
    if isfile(p2)
        p = p2;
        return;
    end
end

p = pathStr;
end

function tf = is_absolute_path(p)
% Windows absolute path: C:\... or \\server\share
% Unix absolute path: /...
tf = (~isempty(regexp(p, '^[A-Za-z]:[\\/]', 'once')) || ...
      ~isempty(regexp(p, '^[\\/]{2}', 'once')) || ...
      startsWith(p, '/'));
end

