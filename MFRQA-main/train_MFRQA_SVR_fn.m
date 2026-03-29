function [model, FeatureTrain] = train_MFRQA_SVR_fn(enhancedPaths, lowlightPaths, mosScores, varargin)
%TRAIN_MFRQA_SVR Train the MFRQA SVR model (LIBSVM epsilon-SVR).
%   [MODEL, FEATURETRAIN] = TRAIN_MFRQA_SVR(ENHANCEDPATHS, LOWLIGHTPATHS, MOSSCORES)
%   extracts MFRQA features for paired images and trains an SVR model.
%
%   Inputs
%   ------
%   enhancedPaths : cell array of file paths to enhanced images
%   lowlightPaths : cell array of file paths to corresponding low-light images
%   mosScores     : numeric vector of subjective quality scores (MOS/DMOS)
%
%   Optional name-value pairs
%   -------------------------
%   'C'         : penalty parameter C (default: 8)
%   'Gamma'     : RBF gamma (default: 0.5)
%   'Epsilon'   : epsilon in epsilon-SVR (default: 0.1)
%   'Verbose'   : true/false to print progress (default: true)
%   'ModelPath' : output .mat file path (default: 'model_mfrqa_svr.mat')
%
%   Notes
%   -----
%   1) This function expects LIBSVM's svmtrain to be available in path.
%   2) The feature extractor assumes image order:
%      img1 = enhanced image, img2 = low-light image.

parser = inputParser;
parser.addParameter('C', 8, @(x) isnumeric(x) && isscalar(x) && x > 0);
parser.addParameter('Gamma', 0.5, @(x) isnumeric(x) && isscalar(x) && x > 0);
parser.addParameter('Epsilon', 0.1, @(x) isnumeric(x) && isscalar(x) && x >= 0);
parser.addParameter('Verbose', true, @(x) islogical(x) || isnumeric(x));
parser.addParameter('ModelPath', 'model_mfrqa_svr.mat', @(x) ischar(x) || isstring(x));
parser.parse(varargin{:});
opts = parser.Results;

if ~iscell(enhancedPaths) || ~iscell(lowlightPaths)
    error('enhancedPaths and lowlightPaths must be cell arrays.');
end

numSamples = numel(mosScores);
if numel(enhancedPaths) ~= numSamples || numel(lowlightPaths) ~= numSamples
    error('Input lengths must match: enhancedPaths, lowlightPaths, mosScores.');
end

mosScores = double(mosScores(:));
FeatureTrain = [];

if opts.Verbose
    fprintf('Extracting features for %d samples...\n', numSamples);
end

for i = 1:numSamples
    imgEnh = imread(enhancedPaths{i});
    imgLow = imread(lowlightPaths{i});

    feat = MFRQA_features(imgEnh, imgLow);
    featRow = feat(:)';

    if i == 1
        FeatureTrain = zeros(numSamples, numel(featRow));
    elseif numel(featRow) ~= size(FeatureTrain, 2)
        error('Feature length mismatch at sample %d: expected %d, got %d.', ...
              i, size(FeatureTrain, 2), numel(featRow));
    end

    FeatureTrain(i, :) = featRow;

    if opts.Verbose && (mod(i, 20) == 0 || i == numSamples)
        fprintf('  %d / %d\n', i, numSamples);
    end
end

svmOptions = sprintf('-s 3 -t 2 -c %g -g %g -p %g -q', opts.C, opts.Gamma, opts.Epsilon);
model = svmtrain(mosScores, FeatureTrain, svmOptions);

modelPath = char(opts.ModelPath);
save(modelPath, 'model');

if opts.Verbose
    fprintf('Training completed. Model saved to: %s\n', modelPath);
end
end
