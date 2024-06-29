import cv2
from tennis_court_detection import is_tennis_court
from player_detection import process_frame, draw_players

def process_video(videoPath):
    cap = cv2.VideoCapture(videoPath)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in range(frame_count):
        ret, frame = cap.read()
        frame = cv2.resize(frame, (900, 500))

        if not ret:
            break

        if is_tennis_court(frame):
            # print(f"Frame {i+1}: Tennis court detected.")
            player_contours = process_frame(frame)
            frame_with_players = draw_players(player_contours, frame)
            cv2.imshow('Player Detection', frame_with_players)
        # else:
        #     print(f"Frame {i+1}: No tennis court detected.")
        #     cv2.imshow('Player Detection', frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        if key == ord('p'):
            cv2.waitKey(-1) #wait until any key is pressed

    cap.release()
    cv2.destroyAllWindows()
