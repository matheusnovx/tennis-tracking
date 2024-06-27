import cv2

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