% DEMO_MODEL_MFRQA_SVR Test script using model_mfrqa_svr.mat
%
% Usage:
%   1) Set imgpath (enhanced image path)
%   2) Set oriimgpath (low-light/reference image path)
%   3) Run this script
%
% Note:
%   This script uses model_mfrqa_svr.mat by default.

clear;
clc;

modelPath = 'model_mfrqa_svr.mat';

if ~exist('imgpath', 'var') || ~exist('oriimgpath', 'var')
    error('Please set imgpath and oriimgpath before running this script.');
end

imgEnh = imread(imgpath);
imgLow = imread(oriimgpath);

score = MFRQA_SVR_with_model(imgEnh, imgLow, modelPath);
fprintf('MFRQA score (model: %s): %.6f\n', modelPath, score);

