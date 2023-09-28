%% GISTCALCULATION.m
% In this file, we load all of our input images and calculate the gist
% descriptor using four scales and six orientations.

% Load all file names
fileNames = erase({dir('~/Geolocation/data/raw/*.jpg').name},".jpg");

% Calculate gist for each image and save it
for i=1:length(fileNames)
   img = rgb2gray(imread(sprintf('~/Geolocation/data/raw/%s.jpg',fileNames{i})));
   data = calcGist(img,4,6);
   save(sprintf("~/Geolocation/data/gistDescriptors2/%s.mat",fileNames{i}),"data");
end
