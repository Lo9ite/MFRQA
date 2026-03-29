function score = MFRQA_SVR_with_model(img1, img2, modelPath)
%MFRQA_SVR_WITH_MODEL Predict quality score using a specified model file.
%
%   score = MFRQA_SVR_with_model(img1, img2, modelPath)
%
% Inputs:
%   img1      - enhanced image
%   img2      - low-light/reference image
%   modelPath - .mat path that contains variable 'model'
%
% Example:
%   score = MFRQA_SVR_with_model(img_enh, img_low, 'model_mfrqa_svr.mat');

if nargin < 3 || isempty(modelPath)
    modelPath = 'model_mfrqa_svr.mat';
end

if ~isfile(modelPath)
    error('Model file not found: %s', modelPath);
end

s = load(modelPath, 'model');
if ~isfield(s, 'model')
    error('Variable ''model'' not found in: %s', modelPath);
end

FeatureTest = MFRQA_features(img1, img2);
[score, ~, ~] = svmpredict(1, FeatureTest, s.model);
end

