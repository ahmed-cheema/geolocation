function gist = calcGist(img,nScales,nOrientations);
% CALCGIST Calculates the gist descriptor of an image
%   Given an input image of size 600x600, computes the gist descriptor 
%   (Oliva and Torralba, 2003) by applying a Gabor filter bank and
%   averaging the filter responses.

    % Set scales
    scales = 2*(1:nScales);

    % Set orientations
    orientations = 0:360/nOrientations:359;

    % Create filter bank
    fb = gabor(scales,orientations);

    % Apply filter bank
    [mag,phase] = imgaborfilt(img,fb);

    % Get 16 grids for each filter response and save average
    idx = 1;
    gist = zeros(1,nScales*nOrientations*16);
    for n = 1:size(phase,3)

        band = phase(:,:,n);

        blocks = zeros(150,150,16);
        c = 1;
        i = 1;
        for b = 1:4
            blocks(:,:,i) = band(c:c+149,1:150);
            i = i+1;
            blocks(:,:,i) = band(c:c+149,151:300);
            i = i+1;
            blocks(:,:,i) = band(c:c+149,301:450);
            i = i+1;
            blocks(:,:,i) = band(c:c+149,451:600);
            i = i+1;
            c = c+149;
        end

        gist(idx:idx+15) = squeeze(mean(blocks,[1 2]));
        idx = idx + 16;
        
    end
