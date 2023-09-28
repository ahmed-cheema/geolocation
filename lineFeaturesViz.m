%% LINEFEATURESVIZ.m
% In this file, we focus on visualizing straight lines for individual
% handpicked images (by choosing n) rather than calcu

%% Loading Data

% Obtain all the image labels in array form
%   Example of image label: grid9_img33
fileNames = erase({dir('~/Geolocation/data/raw/*.jpg').name},".jpg");

%% Analysis

lengthThreshold = .05*600; % Minimum size required for connected components
lineFitThreshold = 0.002; % Minimum "quality of line fit" value required for connected components
nbins = 17; % Number of bins to use in histograms for line angle / length

n = 167;

% Read in image and compute edges matrix with Canny detector
img = imread(sprintf("~/Geolocation/data/raw/%s.jpg",fileNames{n}));
edges = edge(im2gray(img),"canny");

% Find connected components
CC = bwconncomp(edges,8);

% Initialize vectors for line angle and line length
angles = [];
lengths = [];

figure;
imshow(img);
hold on;

% Iterate through connected components (CCs)
for idx = 1:size(CC.PixelIdxList,2)

    % Get indices of values in original image corresponding to this CC
    ind = CC.PixelIdxList{idx};

    % If the size of this CC is less than 5% of the image, discard
    if (length(ind) < lengthThreshold)
        continue
    end

    % Create binary image of just this CC and obtain X & Y coords
    LSR = zeros(size(edges));
    LSR(ind) = 1;
    [ys,xs] = find(LSR == 1);

    % Compute matrix D associated with line support region
    X_tilde = xs-mean(xs);
    Y_tilde = ys-mean(ys);
    XY_tilde_sum = sum(X_tilde.*Y_tilde);
    D = [sum(X_tilde.^2) XY_tilde_sum ;
         XY_tilde_sum sum(Y_tilde.^2)];

    % Compute eigenvalues and eigenvectors of matrix D
    [eigVecs,eigVals] = eig(D);

    % Calculate quality of best fit line
    fit = eigVals(1,1) / eigVals(2,2);

    % If best fit line is over the threshold of 0.01, discard
    if (fit > lineFitThreshold)
        continue
    end
    
    %figure;
    %imshow(LSR);
    %title(fit);
    %pause(0.01);

    % Get eigenvector corresponding to first eigenvalue
    E1 = eigVecs(:,1);

    % Compute line parameters: theta & rho
    theta = atan2(E1(2),E1(1));
    rho = mean(xs)*cos(theta)+mean(ys)*sin(theta);

    % Calculate length
    x = min(xs):max(xs);
    y = (rho - x* cos(theta) )/ sin(theta);
    x1 = x(1);
    y1 = y(1);
    x2 = x(end);
    y2 = y(end);
    dist = sqrt((x2-x1)^2+(y2-y1)^2);
    
    plot([x1 x2], [y1 y2], 'r');

    % Add calculations to vectors
    angles = [angles ; theta];
    lengths = [lengths ; dist];
end

% Compute histogram counts for line angle and line length
%   Range of angle: -pi to pi with 17 bins
%   Range of length: 0 to 850 with 17 bins
angleHist = histcounts(angles,-pi:(2*pi)/nbins:pi);
lengthHist = histcounts(lengths,0:850/nbins:850);

% Create 2x17 matrix with line angle and line length histogram counts
% Save matrix to a .mat file
data = [angleHist ; lengthHist];

%% References
% Code written based on description of methodology from the following
% paper:
%   Košecká, Jana, and Wei Zhang. "Video compass." European conference on computer vision. Springer, Berlin, Heidelberg, 2002.