%% TEXTONHISTOGRAM.m
% In this file, we construct a universal texton dictionary by applying a
% filter bank (a steerable pyramid of four scales and two orientations) to
% 84 images (four from each of 21 grids). For each of these images, the
% eight filter responses are resized if necessary to the input size
% (600x600) and then reshaped into 360000x8 (where each column represents a
% unique filter in the filter bank). This data matrix is computed for all
% 84 images and concatenated to one another - finally, K-means clustering
% is ran on the final data matrix. With our chosen value of K, 32 cluster centers are
% obtained. Then, the aforementioned filter bank is applied to all images in the data set
% and for each image, the 1x8 vector representing filter responses is
% extracted for each pixel and assigned to a cluster. This yields a
% 600x600x1 matrix of cluster labels, which is then used to compute and save a 32
% bin texton histogram for each image.

%% Random Sampling
% Testing has revealed that MATLAB cannot handle running the soon to come
% computations on large data sets. If we want the number of training images
% to be divisible by the number of grids, then 4 is the greatest value of S
% such that the script does not crash when computing with 21*S (the number of grids) 
% images. In order to achieve a representative texton dictionary, we sample
% four images from every grid.

% Load all file names
fileNames = {dir('~/Geolocation/data/raw/*.jpg').name};

% Code adopted from Professor Jerod Weinman Segmentation lab manual
%   https://weinman.cs.grinnell.edu/courses/CSC262/2022F/labs/segmentation.html
rng('default');
rng(42);

nGrid = 21; % Number of grids (pre-determined)
nSample = 4; % Number of samples to take from each grid

% Initialize empty cell array of sampled files
sampleFiles = {};

% Iterate through grids and sample four images from each one
for G = 0:nGrid-1
    % Get files belonging to grid
    subFiles = fileNames(contains(fileNames, sprintf("grid%s_", string(G))));
    
    % Get sample
    sample = datasample(subFiles,nSample);
    
    % Append to cell array
    sampleFiles = [sampleFiles; sample];
end
sampleFiles = sampleFiles(:);
sampleFiles(1)

%% Data Matrix Creation
% 

% Define number of scales and orientations
nScales = 4;
nOrientations = 2;

% Define dimensions of data matrix & initialize it
n = 600*600;
d = nScales*nOrientations;
data = single(zeros(n,d));

% Iterate through images, construct steerable pyramid, resize bands to
% match, and store all band values in data matrix
c = 0;
for i = 1:nGrid*nSample
    % Read grayscale image
    img = rgb2gray(imread(sprintf("~/Geolocation/data/raw/%s",sampleFiles{i})));

    % Build steerable pyramid (converted to single for memory)
    [pyrValues, pyrDims] = buildSFpyr(img, nScales, nOrientations-1);
    pyrValues = single(pyrValues);

    % Iterate through bands of steerable pyramid
    for j = 2:(length(pyrDims)-1)
        % Obtain resized band (converted to single for memory)
        band = single(imresize(pyrBand(pyrValues, pyrDims, j),[600 600]));
        % Save in data matrix
        data(c+1:c+n,j-1) = band(:);
    end
    c = c+n;
end
data = data(1:c,:);

% Clear variables no longer needed in order to free up memory

clear band;
clear pyrValues;
clear pyrDim;
clear img;
clear fileNames;
clear pyrDims;

%% K-Means Clustering
% In this section, we run K-Means Clustering on the data matrix. We seek 32
% clusters.

% Warning: Takes long time to run
K = 32;
[idx,C] = imkmeans(data, K);

%% Cluster Assignments
% Now, we use the previously computed cluster centers to assign pixels in
% all of our images to textons.

% Load all file names
fileNames = erase({dir('~/Geolocation/data/raw/*.jpg').name},".jpg");

for n = 1:length(fileNames)
    
    img = rgb2gray(imread(sprintf("~/Geolocation/data/raw/%s.jpg",fileNames{n})));
    
    % Build steerable pyramid (converted to single for consistency)
    [pyrValues, pyrDims] = buildSFpyr(img, nScales, nOrientations-1);
    pyrValues = single(pyrValues);
    
    % Store resized bands
    bands = zeros(size(img,1),size(img,2),size(C,2));
    for j = 2:(length(pyrDims)-1)
        % Obtain & save resized band (converted to single for consistency)
        bands(:,:,j-1) = single(imresize(pyrBand(pyrValues, pyrDims, j),[size(img,1) size(img,2)]));
    end
    
    % Convert MxNx8 to MNx8
    pxVec = reshape(bands,size(img,1)*size(img,2),d);
    
    % Assign clusters
    clusterLabels = updateAssignments(pxVec,C);
    
    % Create histogram with K bins
    hst = histcounts(clusterLabels,K);
    
    save(sprintf("~/Geolocation/data/textonHistograms/%s.mat",fileNames{n}),"hst");
end
