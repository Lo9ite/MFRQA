function metrics = evaluate_train_pairs_metrics(csvPath)
%EVALUATE_TRAIN_PAIRS_METRICS Compute correlation/error metrics from CSV.
%
%   metrics = evaluate_train_pairs_metrics()
%   metrics = evaluate_train_pairs_metrics(csvPath)
%
% Required columns in CSV:
%   - score
%   - MFRQA
%   - mfrqa_pred_score
%
% Output:
%   metrics: table with rows {MFRQA, mfrqa_pred_score} and columns
%            {PLCC, SRCC, KRCC, RMSE, N}
%
% Notes:
%   - Rows with NaN in either compared column are excluded pairwise.
%   - If the CSV uses "sscore" instead of "score", it will be used as GT.

if nargin < 1 || isempty(csvPath)
    csvPath = 'train_pairs.csv';
end

if ~isfile(csvPath)
    error('CSV file not found: %s', csvPath);
end

T = readtable(csvPath, 'TextType', 'string');

gtCol = '';
if ismember('score', T.Properties.VariableNames)
    gtCol = 'score';
elseif ismember('sscore', T.Properties.VariableNames)
    gtCol = 'sscore';
else
    error('Missing ground-truth column: score (or sscore).');
end

requiredPredCols = {'MFRQA', 'mfrqa_pred_score'};
for i = 1:numel(requiredPredCols)
    if ~ismember(requiredPredCols{i}, T.Properties.VariableNames)
        error('Missing required column: %s', requiredPredCols{i});
    end
end

gt = double(T.(gtCol));
predNames = {'MFRQA', 'mfrqa_pred_score'};

PLCC = nan(2,1);
SRCC = nan(2,1);
KRCC = nan(2,1);
RMSE = nan(2,1);
N = zeros(2,1);

for i = 1:numel(predNames)
    pred = double(T.(predNames{i}));
    valid = ~isnan(gt) & ~isnan(pred);

    x = gt(valid);
    y = pred(valid);
    N(i) = numel(x);

    if N(i) < 2
        warning('Not enough valid samples for %s (N=%d).', predNames{i}, N(i));
        continue;
    end

    PLCC(i) = corr(x, y, 'Type', 'Pearson');
    SRCC(i) = corr(x, y, 'Type', 'Spearman');
    KRCC(i) = corr(x, y, 'Type', 'Kendall');
    RMSE(i) = sqrt(mean((x - y).^2));
end

metrics = table(PLCC, SRCC, KRCC, RMSE, N, 'RowNames', predNames);

fprintf('\nMetrics with GT column: %s\n', gtCol);
disp(metrics);
end

