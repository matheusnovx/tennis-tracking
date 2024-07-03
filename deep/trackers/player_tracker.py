from ultralytics import YOLO
import cv2
import pickle
import sys
import logging

sys.path.append('../')
from utils import measure_distance, get_center_of_bbox

class PlayerTracker:
    """Tracks players in video frames using YOLO model for object detection."""

    def __init__(self, model_path):
        """Initializes the PlayerTracker with a specified YOLO model."""
        self.model = YOLO(model_path)
        logging.basicConfig(level=logging.INFO)

    def choose_and_filter_players(self, court_keypoints, player_detections):
        """Filters player detections based on proximity to court keypoints."""
        if not player_detections:
            logging.warning("No player detections provided.")
            return []

        chosen_player = self.choose_players(court_keypoints, player_detections[0])
        filtered_player_detections = [
            {track_id: bbox for track_id, bbox in player_dict.items() if track_id in chosen_player}
            for player_dict in player_detections
        ]
        return filtered_player_detections

    def choose_players(self, court_keypoints, player_dict):
        """Selects players based on their proximity to court keypoints."""
        if not player_dict:
            logging.warning("Empty player dictionary.")
            return []

        distances = [
            (track_id, min(measure_distance(get_center_of_bbox(bbox), (court_keypoints[i], court_keypoints[i+1]))
                           for i in range(0, len(court_keypoints), 2)))
            for track_id, bbox in player_dict.items()
        ]
        distances.sort(key=lambda x: x[1])
        return [track_id for track_id, _ in distances[:2]]

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        """Detects players in given frames, optionally reading from a stub file."""
        if read_from_stub and stub_path:
            try:
                with open(stub_path, 'rb') as f:
                    return pickle.load(f)
            except IOError as e:
                logging.error(f"Failed to read stub file: {e}")
                return []

        player_detections = [self.detect_frame(frame) for frame in frames]

        if stub_path:
            try:
                with open(stub_path, 'wb') as f:
                    pickle.dump(player_detections, f)
            except IOError as e:
                logging.error(f"Failed to write stub file: {e}")

        return player_detections

    def detect_frame(self, frame):
        """Detects players in a single frame."""
        results = self.model.track(frame, persist=True)[0]
        return {
            int(box.id): box.xyxy[0].tolist()
            for box in results.boxes if results.names[box.cls[0]] == "person"
        }

    def draw_bboxes(self, video_frames, player_detections):
        """Draws bounding boxes around detected players in video frames."""
        for frame, player_dict in zip(video_frames, player_detections):
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = map(int, bbox)
                cv2.putText(frame, f"Player ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        return video_frames