import os
import cv2
import numpy as np
from skimage import feature
import matplotlib.pyplot as plt
import random
from imutils import paths

datapath = "/home/novais/Documents/tennis-tracking/frames"

cell_size = 12

def quantify_image_HoG(image, cell_size=12, visualize=False):
    if visualize:
        features, hog_image = feature.hog(image, orientations=9, 
                                          pixels_per_cell=(cell_size, cell_size),
                                          cells_per_block=(2, 2),
                                          visualize=visualize,
                                          transform_sqrt=True, block_norm="L1")
        return features, hog_image
    else:
        features = feature.hog(image, orientations=9, 
                               pixels_per_cell=(cell_size, cell_size),
                               cells_per_block=(2, 2),
                               transform_sqrt=True, block_norm="L1")
        return features
    
def test_HoG(testingPath, cell_size):
    # Get the list of images
    testingPaths = list(paths.list_images(testingPath))
    
    # Check if testingPaths is empty
    if not testingPaths:
        print(f"No images found in {testingPath}")
        return

    output_images = []

    # Choose 20 random images
    for _ in range(20):
        imagePath = random.choice(testingPaths)

        # Extract the class label from the filename
        label = imagePath.split(os.path.sep)[-2]

        # Print label
        print(f"Processing image: {imagePath} | Label: {label}")
        
        image = cv2.imread(imagePath)
        output = image.copy()
        output = cv2.resize(output, (128, 128))

        # Pre-process the image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (200, 200))
        image = cv2.threshold(image, 0, 255,
                              cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        
        # Quantify the image and make predictions based on the extracted features
        hog, hog_image = quantify_image_HoG(image, cell_size, visualize=True)
        
        # -----------------------------------------
        # Conversion of float to color
        # 1. Multiply by 255
        image = hog_image * 255
        # 2. Truncate to 8-bit integer
        hog_image = image.astype(np.uint8)
        # 3. Convert to color image
        hog_image = cv2.cvtColor(hog_image, cv2.COLOR_GRAY2BGR)
        # ----------------------------------------
        
        # Draw the colored class label on the output image and add it to
        # the set of output images
        color = (255, 255, 0) # if label == "Healthy" else (0, 0, 255)
        cv2.putText(hog_image, label, (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    color, 2)
        output_images.append(hog_image)

    plt.figure(figsize=(20, 20))

    for i in range(len(output_images)):
        plt.subplot(5, 5, i+1)
        plt.imshow(output_images[i])
        plt.axis("off")

    plt.show()

# Verify the constructed path
testingPath = os.path.sep.join(datapath)
print(f"Testing path: {testingPath}")

# Test the function
test_HoG(datapath, cell_size)
