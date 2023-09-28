### PACKAGES ##

from sklearn.metrics.pairwise import haversine_distances
from tensorflow.keras import datasets, layers, models
from math import radians, sin, cos, sqrt, asin
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import cv2 as cv
import pickle
import math
import sys

### HELPER FUNCTIONS & CONSTANTS ####

def progressbar(it, prefix="", size=60, out=sys.stdout):
    """
    Used in for loops to display a progress bar to the user
    Not vital to the project
    Code written by StackOverflow user "imbr"
      https://stackoverflow.com/a/34482761/16924545
    """
    count = len(it)
    def show(j):
        x = int(size*j/count)
        print("{}[{}{}] {}/{}".format(prefix, "#"*x, "."*(size-x), j, count), 
                end='\r', file=out, flush=True)
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    print("\n", flush=True, file=out)

def deg2rad(deg):
    """
    Convert a list of degrees values into radians
    """
    factor = math.pi/180
    return [x * factor for x in deg]

def haversine(lat1, lon1, lat2, lon2):
    """
    Approximate distance between two points (in latitude & longitude)
    using the Haversine formula
      - https://en.wikipedia.org/wiki/Haversine_formula
    """
    # convert degrees to radians
    lat1, lon1, lat2, lon2 = deg2rad([lat1, lon1, lat2, lon2])

    # haversine formula 
    dlat = lat2 - lat1 
    dlon = lon2 - lon1 
    out1 = tf.math.sin(dlat/2)**2 + tf.math.cos(lat1) * tf.math.cos(lat2) * tf.math.sin(dlon/2)**2
    out2 = 2 * tf.math.asin(tf.math.sqrt(out1))  
    return out2 * EARTH_RADIUS_KM

def mean_squared_geodesic_error(y_true, y_pred):
    """
    Calculate the mean squared geodesic error between the true values and the predicted values
    """
    # Extract the latitude and longitude coordinates from the true and predicted values.
    lat1, lon1 = y_true[:, 0], y_true[:, 1]
    lat2, lon2 = y_pred[:, 0], y_pred[:, 1]

    # Calculate the geodesic distance between each pair of true and predicted coordinates.
    err = haversine(lat1, lon1, lat2, lon2)

    # Calculate the mean squared error.
    return tf.reduce_mean(tf.square(err))

def mean_absolute_geodesic_error(y_true, y_pred):
    """
    Calculate the mean squared geodesic error between the true values and the predicted values.
    """
    # Extract the latitude and longitude coordinates from the true and predicted values.
    lat1, lon1 = y_true[:, 0], y_true[:, 1]
    lat2, lon2 = y_pred[:, 0], y_pred[:, 1]

    # Calculate the geodesic distance between each pair of true and predicted coordinates.
    err = haversine(lat1, lon1, lat2, lon2)

    # Calculate the mean absolute error.
    return tf.reduce_mean(tf.math.abs(err))

def getInputs(group):
  """
  Returns a dictionary with the following keys and values based on [group] string input
    - "image": array of images in [group] set
    - "colorHist": array of color histograms in [group] set
    - "gist": array of Gist descriptors in [group] set
    - "textonHist": array of texton histograms in [group] set
    - "lineAngle": array of line angle histograms in [group] set
    - "lineLength": array of line length histograms in [group] set
    - "tinyRGB": array of downscaled (16x16) RGB images in [group] set
    - "tinyLAB": array of downscaled (5x5) LAB images in [group] set
  Valid values of group: "train","validation","test"
  """
  return {"image":images[group],"colorHist":colorHist[0][group],"gist":gist[0][group],
          "textonHist":textonHist[0][group],"lineAngle":lineAngle[0][group],
          "lineLength":lineLength[0][group],"tinyRGB":tinyRGB[0][group],"tinyLAB":tinyLAB[0][group]}

def getNearestPoint(poly,pt):
  """
  Given a point and a polygon, returns nearest point in that polygon from the input point.
  Returns the input point if it is in the polygon.
  """

  # long/lat is reversed
  pt = Point(np.flip(pt))

  # get nearest point
  nearest = nearest_points(poly,pt)
  return np.array((nearest[0].y,nearest[0].x))

# Define radius of earth in kilometers for Haversine distance formula
EARTH_RADIUS_KM = 6371

"""## **Data Loading**"""

### IMPORTING ###

# Load polygon of Spain
espPoly = pickle.load(open("drive/MyDrive/csc262project/espPoly.pkl","rb"))

# Import dictionary DataFrame for all images
db = pd.read_csv("drive/MyDrive/csc262project/overall.csv")

# Load training set images and labels
trainImages = np.load("drive/MyDrive/csc262project/trainImages.npy")
trainLabels = np.load("drive/MyDrive/csc262project/trainLabels.npy")

# Load testing set images and labels
testImages = np.load("drive/MyDrive/csc262project/testImages.npy")
testLabels = np.load("drive/MyDrive/csc262project/testLabels.npy")

# Load validation set images and labels
valImages = np.load("drive/MyDrive/csc262project/valImages.npy")
valLabels = np.load("drive/MyDrive/csc262project/valLabels.npy")

# Get indices for each data set
trainIdx = db[db.group == "train"].index
valIdx = db[db.group == "validation"].index
testIdx = db[db.group == "test"].index

# Load NumPy arrays containing the features for each image
# Reshaping is done to flatten each feature prior to being used in the CNN
# i.e. converting "colorHist" from shape (N,4,14,14) to (N,784)

colorHist = np.load("drive/MyDrive/csc262project/cielabHistograms.npy")
colorHist = colorHist.reshape((db.shape[0],math.prod(colorHist.shape[1:])))

gist = np.load("drive/MyDrive/csc262project/gistDescriptors.npy")
gist = gist.reshape(db.shape[0],math.prod(gist.shape[1:]))

textonHist = np.load("drive/MyDrive/csc262project/textonHistograms.npy")
textonHist = textonHist.reshape(db.shape[0],math.prod(textonHist.shape[1:]))

lineAngle = np.load("drive/MyDrive/csc262project/lineAngles.npy")
lineLength = np.load("drive/MyDrive/csc262project/lineLengths.npy")

tinyRGB = np.load("drive/MyDrive/csc262project/tinyImages.npy")
tinyRGB = tinyRGB.reshape(db.shape[0],math.prod(tinyRGB.shape[1:]))

tinyLAB = np.load("drive/MyDrive/csc262project/tinyLabImages.npy")
tinyLAB = tinyLAB.reshape(db.shape[0],math.prod(tinyLAB.shape[1:]))

"""# **Pre-Processing**

### Data Augmentation
"""

### DATA AUGMENTATION ###
# In this section, I doubled the size of the training set by applying a horizontal flip to each image
# Commented out because I found that augmentation did not improve model performance

"""
import albumentations as albu

transform = albu.HorizontalFlip(always_apply=True,p=1)

trainImagesFlip = np.array([transform(image=x)['image'] for x in trainImages])

trainImages = np.concatenate((trainImages,trainImagesFlip))
trainLabels = np.concatenate((trainLabels,trainLabels))
"""

"""### Normalization"""

### ZERO CENTERING ###
# In this section, I zero centered the input images along each dimension
# I did not convert the images to double (0 to 1 range) due to the greater memory needed to hold float values
# Commented out because I found that zero centering did not improve model performance

"""
means = np.round(np.mean(trainImages, axis=(0,1,2), keepdims=True)).flatten().astype('int16')

meanImage = np.array([np.ones((300,300),dtype='int16')*m for m in means]).reshape((300,300,3))

trainImages = np.subtract(trainImages,meanImage,casting="same_kind")
testImages = np.subtract(testImages,meanImage,casting="same_kind")
valImages = np.subtract(valImages,meanImage,casting="same_kind")
"""

"""# **Features**

### Feature Normalization
"""

### FEATURE NORMALIZATION ###

# Define dictionairies of each feature, splitting by the set they belong to
features = [colorHist,gist,textonHist,lineAngle,lineLength,tinyRGB,tinyLAB]
featureDicts = []
for f in features:
  # normalize
  f = np.true_divide(f,np.max(f),casting="same_kind")

  # zero center
  f -= f.mean(axis=1, keepdims=True)

  featureDicts.append({"train":f[trainIdx],
                       "validation":f[valIdx],
                       "test":f[testIdx]})

# Redefine features as their normalized versions  
colorHist,gist,textonHist,lineAngle,lineLength,tinyRGB,tinyLAB = zip(featureDicts)

# Now, all the color histograms for the training set can simply be called with:
#   > colorHist["train"]

# For consistency, use same data structure for images
images = {"train":trainImages,
          "validation":valImages,
          "test":testImages}

### RANDOM PREDICTION ###

# Combine train and validation labels - no need for a separate validation set
fullTrainLabels = np.concatenate((trainLabels,valLabels))

# Make random predictions
errRand = []
for i in progressbar(range(len(testIdx)),"Computing: ",20):
  queryPt = testLabels[i]
  randPred = fullTrainLabels[np.random.randint(fullTrainLabels.shape[0], size=1), :]
  errRand.append(np.max(haversine_distances([[radians(v) for v in queryPt],[radians(w) for w in randPred[0]]]))*EARTH_RADIUS_KM)

# Get model diagnostics
print("Mean Geodesic Error: {:.2f}".format(np.mean(errRand)))
print("Mean Squared Geodesic Error: {:.2f}".format(np.mean(np.square(errRand))))
print("Pct of Predictions within 50 km: {:.2%}".format(len([x for x in errRand if x < 50])/len(errRand)))
print("Pct of Predictions within 100 km: {:.2%}".format(len([x for x in errRand if x < 100])/len(errRand)))
print("Pct of Predictions within 200 km: {:.2%}".format(len([x for x in errRand if x < 200])/len(errRand)))
print("Pct of Predictions within 300 km: {:.2%}".format(len([x for x in errRand if x < 300])/len(errRand)))
print("Pct of Praedictions within 400 km: {:.2%}".format(len([x for x in errRand if x < 400])/len(errRand)))
print("Pct of Predictions within 500 km: {:.2%}".format(len([x for x in errRand if x < 500])/len(errRand)))
print("Pct of Predictions within 600 km: {:.2%}".format(len([x for x in errRand if x < 600])/len(errRand)))

"""# **Convolutional Neural Networks**

## **No Features**

### Model Definition
"""

model = models.Sequential()

model.add(layers.Conv2D(64, 7, strides=2, padding="same", input_shape=(300,300,3), activation="relu"))
model.add(layers.MaxPool2D(3, strides=2))

model.add(layers.Conv2D(64, 5, padding="same", activation="relu"))
model.add(layers.Conv2D(192, 3, padding="same", activation="relu"))
model.add(layers.MaxPool2D(3, strides=2))

model.add(layers.Conv2D(128, 1, padding="same", activation="relu"))
model.add(layers.Conv2D(256, 3, padding="same", activation="relu"))
model.add(layers.MaxPool2D(3, strides=2))

model.add(layers.Conv2D(192, 1, padding="same", activation="relu"))
model.add(layers.Conv2D(192, 3, padding="same", activation="relu"))
model.add(layers.MaxPool2D(3, strides=2))

model.add(layers.Flatten())

model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(32, activation="relu"))
model.add(layers.Dense(2, activation="linear"))

model.summary()

"""### Model Compile & Fit"""

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss=mean_squared_geodesic_error,
              metrics=['mse'])

history = model.fit(trainImages, trainLabels, batch_size=32, epochs=20, verbose=2,
                     validation_data=(valImages, valLabels))

"""### Predictions"""

### PREDICTIONS ###
# In this section, the model will be used for predictions and these predictions are analyzed

# Predict location on test images
pred = model.predict(testImages)

# Clip predicted locations outside of Spain
pred = [getNearestPoint(espPoly,p) for p in pred]

# Plot predictions
x,y = espPoly.exterior.xy
plt.plot(x,y,color='black',linewidth=0.5)
plt.scatter([x[1] for x in pred],[x[0] for x in pred],edgecolor='black',alpha=0.3,color='orange')
plt.xlabel("Latitude")
plt.ylabel("Longitude")
plt.title("Predictions for CNN without Features")
plt.show()

# Plot distribution of errors
err = [np.max(haversine_distances([[radians(v) for v in x],[radians(w) for w in y]]))*EARTH_RADIUS_KM for x,y in zip(pred,testLabels)]
plt.hist(err,bins=20,edgecolor='black')
plt.xlabel("Error (km)")
plt.ylabel("Frequency")
plt.title("Error Distribution for CNN without Features")
plt.show()

# Get model diagnostics
print("Mean Geodesic Error: {:.2f}".format(np.mean(err)))
print("Mean Squared Geodesic Error: {:.2f}".format(np.mean(np.square(err))))
print("Pct of Predictions within 50 km: {:.2%}".format(len([x for x in err if x < 50])/len(err)))
print("Pct of Predictions within 100 km: {:.2%}".format(len([x for x in err if x < 100])/len(err)))
print("Pct of Predictions within 200 km: {:.2%}".format(len([x for x in err if x < 200])/len(err)))
print("Pct of Predictions within 300 km: {:.2%}".format(len([x for x in err if x < 300])/len(err)))
print("Pct of Praedictions within 400 km: {:.2%}".format(len([x for x in err if x < 400])/len(err)))
print("Pct of Predictions within 500 km: {:.2%}".format(len([x for x in err if x < 500])/len(err)))
print("Pct of Predictions within 600 km: {:.2%}".format(len([x for x in err if x < 600])/len(err)))

model.save("model")

"""## **With Features**

### Model Definition
"""

# Define separate inputs: the image and the features
inputs = [tf.keras.Input(shape=(300,300,3), name="image"), 
          tf.keras.Input(shape=(784,), name="colorHist"),
          tf.keras.Input(shape=(384,), name="gist"),
          tf.keras.Input(shape=(32,), name="textonHist"),
          tf.keras.Input(shape=(17,), name="lineAngle"),
          tf.keras.Input(shape=(17,), name="lineLength"),
          tf.keras.Input(shape=(768,), name="tinyRGB"),
          tf.keras.Input(shape=(75,), name="tinyLAB")]

### CONVOLUTION STAGE ###

# Only the image goes through these convolution / pooling layers

conv1 = layers.Conv2D(64, 7, strides=2, padding="same", activation="relu")(inputs[0])
pool1 = layers.MaxPool2D(3, strides=2)(conv1)

conv2 = layers.Conv2D(64, 5, padding="same", activation="relu")(pool1)
conv3 = layers.Conv2D(192, 3, padding="same", activation="relu")(conv2)
pool2 = layers.MaxPool2D(3, strides=2)(conv3)

conv4 = layers.Conv2D(128, 1, padding="same", activation="relu")(pool2)
conv5 = layers.Conv2D(256, 3, padding="same", activation="relu")(conv4)
pool3 = layers.MaxPool2D(3, strides=2)(conv5)

conv6 = layers.Conv2D(192, 1, padding="same", activation="relu")(pool3)
conv7 = layers.Conv2D(192, 3, padding="same", activation="relu")(conv6)
pool4 = layers.MaxPool2D(3, strides=2)(conv7)

### FLATTENING -> FULLY CONNECTED LAYER ###

# Flatten the output of the image passing through the convolution stage
# Pass through Dense layer with 128 units
flat = layers.Flatten()(pool4)
imgDense = layers.Dense(128, activation="relu")(flat)

# Now, pass each of the features through two Dense layers with 128 units:

colorHistDense = layers.Dense(128, activation="relu")(inputs[1])
colorHistDense = layers.Dense(128, activation="relu")(colorHistDense)

gistDense = layers.Dense(128, activation="relu")(inputs[2])
gistDense = layers.Dense(128, activation="relu")(gistDense)

textonHistDense = layers.Dense(128, activation="relu")(inputs[3])
textonHistDense = layers.Dense(128, activation="relu")(textonHistDense)

lineAngleDense = layers.Dense(128, activation="relu")(inputs[4])
lineAngleDense = layers.Dense(128, activation="relu")(lineAngleDense)

lineLengthDense = layers.Dense(128, activation="relu")(inputs[5])
lineLengthDense = layers.Dense(128, activation="relu")(lineLengthDense)

tinyRGBDense = layers.Dense(128, activation="relu")(inputs[6])
tinyRGBDense = layers.Dense(128, activation="relu")(tinyRGBDense)

tinyLABDense = layers.Dense(128, activation="relu")(inputs[7])
tinyLABDense = layers.Dense(128, activation="relu")(tinyLABDense)

# Concatenate all eight (one for image, seven for features) outputs together

concat = layers.Concatenate()([imgDense,
                               colorHistDense,
                               gistDense,
                               textonHistDense,
                               lineAngleDense,
                               lineLengthDense,
                               tinyRGBDense,
                               tinyLABDense
                               ])

# Pass through final Dense layers to make prediction
outputs = layers.Dense(128, activation="relu")(concat)
outputs = layers.Dense(128, activation="relu")(outputs)
outputs = layers.Dense(2, activation="linear")(outputs)

modelFeat = tf.keras.Model(inputs=inputs, outputs=outputs)

modelFeat.summary()

"""### Model Compile & Fit"""

modelFeat.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss=mean_squared_geodesic_error,
                  metrics=['mse'])

historyFeat = modelFeat.fit(getInputs("train"), trainLabels, batch_size=32, epochs=20, verbose=2,
                            validation_data=(getInputs("validation"), valLabels))

"""### Predictions"""

### PREDICTIONS ###
# In this section, the model will be used for predictions and these predictions are analyzed

# Predict location on test images
predFeat = modelFeat.predict(getInputs("test"))

# Clip predicted locations outside of Spain
predFeat = [getNearestPoint(espPoly,p) for p in predFeat]

# Plot predictions
x,y = espPoly.exterior.xy
plt.plot(x,y,color='black',linewidth=0.5)
plt.scatter([x[1] for x in predFeat],[x[0] for x in predFeat],edgecolor='black',alpha=0.3,color='orange')
plt.xlabel("Latitude")
plt.ylabel("Longitude")
plt.title("Predictions for CNN with Features")
plt.show()

# Plot distribution of errors
errFeat = [np.max(haversine_distances([[radians(v) for v in x],[radians(w) for w in y]]))*EARTH_RADIUS_KM for x,y in zip(predFeat,testLabels)]
plt.hist(errFeat,bins=20,edgecolor='black')
plt.xlabel("Error (km)")
plt.ylabel("Frequency")
plt.title("Error Distribution for CNN with Features")
plt.show()

# Get model diagnostics
print("Mean Geodesic Error: {:.2f}".format(np.mean(errFeat)))
print("Mean Squared Geodesic Error: {:.2f}".format(np.mean(np.square(errFeat))))
print("Pct of Predictions within 50 km: {:.2%}".format(len([x for x in errFeat if x < 50])/len(errFeat)))
print("Pct of Predictions within 100 km: {:.2%}".format(len([x for x in errFeat if x < 100])/len(errFeat)))
print("Pct of Predictions within 200 km: {:.2%}".format(len([x for x in errFeat if x < 200])/len(errFeat)))
print("Pct of Predictions within 300 km: {:.2%}".format(len([x for x in errFeat if x < 300])/len(errFeat)))
print("Pct of Praedictions within 400 km: {:.2%}".format(len([x for x in errFeat if x < 400])/len(errFeat)))
print("Pct of Predictions within 500 km: {:.2%}".format(len([x for x in errFeat if x < 500])/len(errFeat)))
print("Pct of Predictions within 600 km: {:.2%}".format(len([x for x in errFeat if x < 600])/len(errFeat)))

modelFeat.save("modelFeat")