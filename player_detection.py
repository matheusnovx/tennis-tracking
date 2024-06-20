import cv2
import court_detection
import extract_train

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

        # Process the frame to detect players
        # player_contours = process_frame(frame)
        
        # Draw players on the frame
        #frame_with_players = draw_players(player_contours, frame)
        
        # Display the frame with player detection
        #cv2.imshow('Player Detection', frame_with_players)

        cv2.imshow('Player Detection', frame)

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
