import cv2
import numpy as np
import joblib

# Function to extract HOG features
def extract_hog_features(image, size=(128, 128)):
    hog = cv2.HOGDescriptor()
    image = cv2.resize(image, size)
    h = hog.compute(image)
    return h

# Function to check if a tennis court is in the frame
def is_tennis_court(frame, classifier, scaler):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_frame = cv2.resize(gray, (128, 128))
    hog_features = extract_hog_features(resized_frame).reshape(1, -1)
    hog_features = scaler.transform(hog_features)
    prediction = classifier.predict(hog_features)
    return prediction[0] == 1

def main(video_path, classifier_path, scaler_path):
    classifier = joblib.load(classifier_path)
    scaler = joblib.load(scaler_path)
    
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if is_tennis_court(frame, classifier, scaler):
            print("Tennis court detected in the frame")
            # Process the frame further if needed
        else:
            print("No tennis court detected in the frame")
        
        cv2.imshow('Frame', frame)
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Path to the input video and classifier
video_path = 'tennis_match2.mp4'
classifier_path = 'tennis_court_classifier.pkl'
scaler_path = 'scaler.pkl'

# Run the main function
main(video_path, classifier_path, scaler_path)
