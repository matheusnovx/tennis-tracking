from typing import Tuple
import numpy as np

def get_center_of_bbox(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
    """
    Calculate the center point of a bounding box.
    
    Parameters:
    - bbox: A tuple of (x1, y1, x2, y2) representing the bounding box.
    
    Returns:
    - A tuple of (center_x, center_y) representing the center of the bounding box.
    """
    bbox_array = np.array(bbox)
    center = (bbox_array[:2] + bbox_array[2:]) // 2
    
    return tuple(center)

def measure_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    """
    Measure the Euclidean distance between two points.
    
    Parameters:
    - p1, p2: Tuples of (x, y) representing the points.
    
    Returns:
    - The Euclidean distance between p1 and p2.
    """
    return np.linalg.norm(np.array(p1) - np.array(p2))