import os
import cv2
import numpy as np
from skimage import feature
import random
from imutils import paths
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

datapath = "/home/novais/Documents/tennis-tracking/frames"
cell_size = 12

def quantify_image_HoG(image, cell_size=12):
    features = feature.hog(image, orientations=9, 
                           pixels_per_cell=(cell_size, cell_size),
                           cells_per_block=(2, 2),
                           transform_sqrt=True, block_norm="L1")
    return features

# Prepare dataset
def load_dataset(datasetPath):
    imagePaths = list(paths.list_images(datasetPath))
    data = []
    labels = []

    for imagePath in imagePaths:
        label = imagePath.split(os.path.sep)[-2]
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (200, 200))
        image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        features = quantify_image_HoG(image, cell_size)
        data.append(features)
        labels.append(label)

    return np.array(data), np.array(labels)

# Load the dataset
print("[INFO] loading data...")
data, labels = load_dataset(datapath)

# Encode the labels
le = LabelEncoder()
labels = le.fit_transform(labels)

# Split the data into training and testing sets
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

# Train the classifier
print("[INFO] training classifier...")
model = LinearSVC()
model.fit(trainX, trainY)

# Evaluate the classifier
print("[INFO] evaluating classifier...")
predictions = model.predict(testX)
print(classification_report(testY, predictions, target_names=le.classes_))

# Save the model and label encoder
import pickle

with open('model.pickle', 'wb') as f:
    pickle.dump(model, f)
    
with open('label_encoder.pickle', 'wb') as f:
    pickle.dump(le, f)
