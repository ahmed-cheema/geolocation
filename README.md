# Small-Scale Photo Geolocation with Feature-Based Convolutional Neural Networks

MATLAB was used for calculating the following three features. Afterwards, Python is used for the remainder of this project.

## Line Features
- The folder "lineFeaturesCalc" contains the relevant code for line features.
- The file "lineFeatures.m" contains a for-loop that iterates over all images in a given directory and saves the histograms representing line statistics for each image.
- The file "lineFeaturesViz.m" is an adapted version that is used for visualizing straight lines.

## Gist Descriptor
- The folder "gistDescriptorCalc" contains the relevant code for Gist descriptors.
- The file "calcGist.m" contains the code for a function that calculates the gist descriptor of an image.
- The file "gistCalculation.m" is similar to "lineFeatures.m" - it contains a for-loop that iterates over all images in a given directory and saves the gist descrptor for each image.

## Texton Histogram
- The folder "textonHistogramCalc" contains the relevant code for texton histograms.
- The file "textonHistogram.m" contains code for constructing a texton dictionary and creating texton histograms for each image.
- The "imkmeans.m", "updateAssignments.m", "updateCenters.n" files include functions from the Segmentation lab for k-means clustering

Once the ".m" files are used to calculate texton histograms, line features, and gist descriptors, the following Python files should be run (in order):

**preprocessing.ipynb** - contains code for splitting data based on source

**features.ipynb** - calculate remaining features, run 1-NN feature predictions

**preparation.ipynb** - splits into validation set, saves arrays used for final modeling

**modeling.py** - contains convolutional neural network code
