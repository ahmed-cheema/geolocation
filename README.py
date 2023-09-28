# Once the ".m" files are used to calculate texton histograms, line features, and gist descriptors, the following Python files should be run (in order):

# preprocessing.ipynb - contains code for splitting data based on source
# features.ipynb - calculate remaining features, run 1-NN feature predictions
# preparation.ipynb - splits into validation set, saves arrays used for final modeling
# modeling.py - contains convolutional neural network code