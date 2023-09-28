% Function signature from Prof. Weinman's Segmentation lab manual
%   https://weinman.cs.grinnell.edu/courses/CSC262/2022F/labs/segmentation.html
% Code adapted from Ahmed Cheema's solution to Prof. Weinman's Segmentation
% lab
function centers = updateCenters(data, labels, K);
% UPDATECENTERS Updates cluster centers based on inputted raw data and
% cluster assignments
%   Given an input data matrix (of size MNx3 for an MxNx3 RGB image), an
%   MNx1 vector of cluster assignments, and a scalar K indicating the
%   number of clusters, outputs newly calculated cluster centers based on
%   the mean color found for each cluster's assigned pixels.

    centers = zeros(K,size(data,2));

    for (currentK = 1:K)
        [ind] = find(labels == currentK);

        if (size(ind,1) == 0)
            centers(currentK,:) = rand(1,3);
        end

        centers(currentK,:) = mean(data(ind,:));
    end
end

