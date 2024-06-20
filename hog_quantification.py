import numpy as np
from skimage import feature

def quantify_image_HoG(image, cell_size=12):
    features = feature.hog(image, orientations=9, 
                           pixels_per_cell=(cell_size, cell_size),
                           cells_per_block=(2, 2),
                           transform_sqrt=True, block_norm="L1")
    return features
