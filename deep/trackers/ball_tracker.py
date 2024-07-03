from ultralytics import YOLO 
import cv2
import pickle
import pandas as pd

class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def _create_dataframe_from_positions(self, ball_positions):
        """Convert ball positions to a pandas DataFrame."""
        positions = [x.get(1, []) for x in ball_positions]
        return pd.DataFrame(positions, columns=['x1', 'y1', 'x2', 'y2'])

    def interpolate_ball_positions(self, ball_positions):
        df_ball_positions = self._create_dataframe_from_positions(ball_positions)
        df_ball_positions.interpolate(inplace=True)
        df_ball_positions.bfill(inplace=True)
        return [{1: x} for x in df_ball_positions.to_numpy().tolist()]

    def get_ball_shot_frames(self, ball_positions):
        df_ball_positions = self._create_dataframe_from_positions(ball_positions)
        df_ball_positions['ball_hit'] = 0
        df_ball_positions['mid_y'] = (df_ball_positions['y1'] + df_ball_positions['y2']) / 2
        df_ball_positions['mid_y_rolling_mean'] = df_ball_positions['mid_y'].rolling(window=5, min_periods=1).mean()
        df_ball_positions['delta_y'] = df_ball_positions['mid_y_rolling_mean'].diff()

        # Detect ball hits with vectorized operations
        df_ball_positions['change_sign'] = df_ball_positions['delta_y'].diff().lt(0)
        hits = df_ball_positions['change_sign'].rolling(window=25, min_periods=1).sum()
        df_ball_positions.loc[hits > 20, 'ball_hit'] = 1

        return df_ball_positions[df_ball_positions['ball_hit'] == 1].index.tolist()

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path:
            try:
                with open(stub_path, 'rb') as f:
                    return pickle.load(f)
            except IOError:
                print("Error reading stub file.")

        ball_detections = [self.detect_frame(frame) for frame in frames]

        if stub_path:
            try:
                with open(stub_path, 'wb') as f:
                    pickle.dump(ball_detections, f)
            except IOError:
                print("Error saving stub file.")

        return ball_detections

    def detect_frame(self, frame):
        try:
            results = self.model.predict(frame, conf=0.15)[0]
            ball_dict = {1: box.xyxy.tolist()[0] for box in results.boxes}
            return ball_dict
        except Exception as e:
            print(f"Error in model prediction: {e}")
            return {}

    def draw_bboxes(self, video_frames, player_detections):
        for frame, ball_dict in zip(video_frames, player_detections):
            for track_id, bbox in ball_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Ball ID: {track_id}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
        return video_frames