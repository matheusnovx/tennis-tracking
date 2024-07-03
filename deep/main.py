import logging
from typing import List, Tuple
from utils import (read_video, 
                   save_video)
from trackers import PlayerTracker, BallTracker
from court_line_detection import CourtLineDetector

# Initialize player, ball trackers, and court line detector
def initialize_trackers() -> Tuple[PlayerTracker, BallTracker, CourtLineDetector]:
    player_tracker = PlayerTracker(model_path="deep/models/yolov8x.pt")
    ball_tracker = BallTracker(model_path="deep/models/yolo5_last.pt")
    court_line_detector = CourtLineDetector(model_path="deep/models/keypoints_model.pth")
    return player_tracker, ball_tracker, court_line_detector

# Detect players and the ball in the video
def detect_objects(video_frames: List, player_tracker: PlayerTracker, ball_tracker: BallTracker) -> Tuple[List, List]:
    player_detections = player_tracker.detect_frames(video_frames, read_from_stub=True, stub_path="deep/pickle/player_detections.pkl")
    ball_detections = ball_tracker.detect_frames(video_frames, read_from_stub=True, stub_path="deep/pickle/ball_detections.pkl")
    return player_detections, ball_detections

# Process the video
def process_video(input_video_path: str):
    video_frames = read_video(input_video_path)
    player_tracker, ball_tracker, court_line_detector = initialize_trackers()
    
    court_keypoints = court_line_detector.predict(video_frames[0])
    
    player_detections, ball_detections = detect_objects(video_frames, player_tracker, ball_tracker)
    player_detections = player_tracker.choose_and_filter_players(court_keypoints, player_detections)
    
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)
    
    save_video(output_video_frames, "output_videos/output_deep.avi")

def main():
    input_video_path = "input_videos/tennis_match2_cut.mp4"
    process_video(input_video_path)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()