import numpy as np
from dataset_loader import load_dataset
from model_training import train_and_save_model
from video_processing import process_video, read_video, save_video

# Path to the dataset and cell size for HOG
datapath = "classic/frames"
cell_size = 12

# Load the dataset
print("[INFO] loading data...")
data, labels = load_dataset(datapath, cell_size)

# Check the distribution of classes
unique, counts = np.unique(labels, return_counts=True)
class_distribution = dict(zip(unique, counts))
print(f"[INFO] class distribution: {class_distribution}")

# Train the model and save it
train_and_save_model(data, labels)

# Path to the input video
video_path = 'input_videos/tennis_match2_cut.mp4'

# Process the video
# process_video(video_path)
video_frames = read_video(video_path)
save_video(video_frames, "output_videos/output_classic.avi")
