import cv2
from tennis_court_detection import is_tennis_court
from player_detection import process_frame, draw_players

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    i = 1
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if i % 10 == 0:
            print(f"Processing frame {i} of {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}...")

        if is_tennis_court(frame):
            player_contours = process_frame(frame)
            frame_with_players = draw_players(player_contours, frame)
            frames.append(frame_with_players)
            i += 1

    cap.release()
    return frames

def save_video(frames, output_path):
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_path, fourcc, 24, (frames[0].shape[1], frames[0].shape[0]))
    for frame in frames:
        out.write(frame)
    out.release()

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
