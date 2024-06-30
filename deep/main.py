from utils import (read_video, 
                   save_video)
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt
import cv2

def main():
    input_video_path = "input_videos/tennis_match2_cut.mp4"
    video_frames = read_video(input_video_path)

    player_tracker = PlayerTracker(model_path="yolov8x")
    ball_tracker = BallTracker(model_path="deep/models/yolo5_last.pt")

    player_detections = player_tracker.detect_frames(video_frames,
                                                     read_from_stub=True,
                                                     stub_path="deep/tracker_stubs/player_detections.pkl")
    

    ball_detections = ball_tracker.detect_frames(video_frames,
                                                     read_from_stub=True,
                                                     stub_path="deep/tracker_stubs/ball_detections.pkl")

    court_model_path = "deep/models/keypoints_model.pth"
    court_line_detector = CourtLineDetector(model_path=court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[0])

    mini_court = MiniCourt(video_frames[0])

    player_detections = player_tracker.choose_and_filter_players(court_keypoints, player_detections)


    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)
    output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)
    output_video_frames = mini_court.draw_mini_court(output_video_frames)

    save_video(output_video_frames, "output_videos/output.avi")

if __name__ == "__main__":
    main()