% Function signature from Prof. Weinman's Segmentation lab manual
%   https://weinman.cs.grinnell.edu/courses/CSC262/2022F/labs/segmentation.html
% Code adapted from Ahmed Cheema's solution to Prof. Weinman's Segmentation
% lab
function labels = updateAssignments(data, centers);
% UPDATEASSIGNMENTS Updates the pixel assignment to clusters based on
% Euclidean distance
%   Given an input data matrix of size MNxC for an image (of size MxNx3 if an RGB image) and a
%   matrix of cluster centers of size KxC, returns MNx1 vector of cluster
%   assignments for each individual pixel. The cluster assignment for each
%   individual pixel is equal to the cluster center for which the Euclidean
%   distance (sum squared distance across C channels) is minimized.

    K = size(centers,1);

    labels = zeros(size(data,1),1);
    closestClusterDist = Inf(size(data,1),1);

    for (currentK = 1:K)

        thisClusterDist = sum((data-centers(currentK,:)).^2,2);

        ind = find(thisClusterDist < closestClusterDist);

        labels(ind) = currentK;
        closestClusterDist(ind) = thisClusterDist(ind);
    end
end

