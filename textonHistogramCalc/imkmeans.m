% Function signature from Prof. Weinman's Segmentation lab manual
%   https://weinman.cs.grinnell.edu/courses/CSC262/2022F/labs/segmentation.html
% Code adapted from Ahmed Cheemas solution to Prof. Weinman's Segmentation
% lab
function [labels, centers] = imkmeans( data, K )
%IMKMEANS Categorizes pixels in an inputted matrix into K clusters
%   Given an image of size MxNxC (in the case of RGB images, C=3), 
%   categorizes all pixels into one of K clusters using an algorithm
%   based on minimizing Euclidean distance. Outputs an MNx1 vector of
%   cluster assignments and a KxC matrix of final cluster centers.
    %rng('default');
    %rng(42);
    
    randCenters = rand(K,size(data,2));
    
    labels = updateAssignments(data,randCenters);
    
    for (i = 1:100)
        centers = updateCenters(data, labels, K);
        newLabels = updateAssignments(data, centers);
        if (all(newLabels == labels))
            %disp("max reached");
            break
        end
        labels = newLabels;
    end
end

