function [model, FeatureTrain] = train_MFRQA_SVR(enhancedPaths, lowlightPaths, mosScores, varargin)
%TRAIN_MFRQA_SVR Backward-compatible wrapper for train_MFRQA_SVR_fn.
%
% This wrapper keeps the original entrypoint name while delegating to
% train_MFRQA_SVR_fn, which avoids conflicts when a local script with the
% same name exists in MATLAB path.

[model, FeatureTrain] = train_MFRQA_SVR_fn(enhancedPaths, lowlightPaths, mosScores, varargin{:});
end
