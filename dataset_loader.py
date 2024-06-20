import os
import cv2
import numpy as np
from imutils import paths
from hog_quantification import quantify_image_HoG

def load_dataset(datasetPath, cell_size=12):
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
