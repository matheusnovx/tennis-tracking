import os
import cv2
import numpy as np
from skimage import feature
from imutils import paths
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle

# Path to the dataset and cell size for HOG
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

# Check the distribution of classes
unique, counts = np.unique(labels, return_counts=True)
class_distribution = dict(zip(unique, counts))
print(f"[INFO] class distribution: {class_distribution}")

# Encode the labels
le = LabelEncoder()
labels = le.fit_transform(labels)

# Check if we have at least two classes
if len(np.unique(labels)) < 2:
    raise ValueError("The dataset must contain at least two classes.")

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
with open('model.pickle', 'wb') as f:
    pickle.dump(model, f)
    
with open('label_encoder.pickle', 'wb') as f:
    pickle.dump(le, f)

def is_tennis_court(image, cell_size=12):
    with open('model.pickle', 'rb') as f:
        model = pickle.load(f)
    
    with open('label_encoder.pickle', 'rb') as f:
        le = pickle.load(f)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (200, 200))
    image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    features = quantify_image_HoG(image, cell_size)

    prediction = model.predict([features])
    label = le.inverse_transform(prediction)[0]
    
    return label == "tennis"

# Process video to detect tennis court in each frame
def process_video(videoPath):
    cap = cv2.VideoCapture(videoPath)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        if is_tennis_court(frame):
            print(f"Frame {i+1}: Tennis court detected.")
        else:
            print(f"Frame {i+1}: No tennis court detected.")

    cap.release()

# Background subtractor for player detection
background_subtractor = cv2.createBackgroundSubtractorMOG2()

def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply background subtraction
    fg_mask = background_subtractor.apply(gray_blurred)
    
    # Apply morphological operations to clean the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area size to focus on players
    player_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 500]  # Adjust the threshold as needed
    
    return player_contours

def draw_players(player_contours, frame):
    for contour in player_contours:
        # Get the bounding box for each player contour
        x, y, w, h = cv2.boundingRect(contour)
        
        # Draw the detected player
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    return frame

def main(video_path):
    # Open video capture
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.resize(frame, (900, 500))   

        if not ret:
            break

        # Check if the frame contains a tennis court
        if is_tennis_court(frame):
            print("Tennis court detected.")
            # Process the frame to detect players
            player_contours = process_frame(frame)
            # Draw players on the frame
            frame_with_players = draw_players(player_contours, frame)
            cv2.imshow('Player Detection', frame_with_players)
        else:
            print("No tennis court detected.")

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        if key == ord('p'):
            cv2.waitKey(-1) #wait until any key is pressed
    
    cap.release()
    cv2.destroyAllWindows()

# Path to the input video
video_path = 'tennis_match2.mp4'

# Run the main function
main(video_path)
