import mediapipe as mp
import cv2
import numpy as np
import os
import logging
import sys
import warnings
import tensorflow as tf

tf.get_logger().setLevel('ERROR')

# Suppress Mediapipe and protobuf warnings
os.environ['GLOG_minloglevel'] = '2'
logging.getLogger('mediapipe').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf')

# Mediapipe pose models
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Improved video path handling
output_folder = r"file path"
output_video_path = os.path.join(output_folder, "file name")

# Create output directory if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Open video
input_video_path = r""file path""
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get the original video properties (frame rate, resolution)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Video properties: FPS = {fps}, Width = {width}, Height = {height}")

# Define the output video codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
output_width, output_height = 1280,720  # Optimal resolution for body tracking
out = cv2.VideoWriter(output_video_path, fourcc, fps, (output_width, output_height))

# Ensure VideoWriter object is opened successfully
if not out.isOpened():
    print("Error: Could not open the output video file for writing.")
    exit()

frame_count = 0

# Define drawing specifications for smoother visuals
landmark_drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3)
connection_drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)

# List of pose landmarks to keep (excluding face and hands)
body_keypoints_indices = [
    11, 12,  # Shoulders
    13, 14,  # Elbows
    15, 16,  # Wrists (singular wrist point)
    23, 24,  # Hips
    25, 26,  # Knees
    27, 28,  # Ankles
    31, 32   # Feet
]

# Initiate pose model
with mp_pose.Pose(static_image_mode=False, model_complexity=2, min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame for faster processing and better quality
        frame = cv2.resize(frame, (output_width, output_height), interpolation=cv2.INTER_LINEAR)

        # Recolor feed to RGB for Mediapipe processing
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Make detections
        results = pose.process(image)

        # Recolor image back to BGR for rendering
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw pose landmarks with customized appearance (excluding face, hands, and shoulder line)
        if results.pose_landmarks:
            # Draw only the specific body keypoints (excluding face and hands)
            for idx in body_keypoints_indices:
                if idx < len(results.pose_landmarks.landmark):
                    landmark = results.pose_landmarks.landmark[idx]
                    x = int(landmark.x * image.shape[1])
                    y = int(landmark.y * image.shape[0])
                    cv2.circle(image, (x, y), 5, (0, 255, 0), -1)  # Draw body keypoints in green

            # Draw the connections only for body parts (excluding face and hands)
            body_connections = [
                (11, 13), (13, 15),  # Left arm
                (12, 14), (14, 16),  # Right arm
                (11, 23), (12, 24),  # Shoulders to hips
                (23, 25), (25, 27),  # Left leg
                (24, 26), (26, 28),  # Right leg
                (27, 31), (28, 32)   # Ankles to feet
            ]

            for start_idx, end_idx in body_connections:
                if start_idx < len(results.pose_landmarks.landmark) and end_idx < len(results.pose_landmarks.landmark):
                    start_point = (int(results.pose_landmarks.landmark[start_idx].x * image.shape[1]),
                                   int(results.pose_landmarks.landmark[start_idx].y * image.shape[0]))
                    end_point = (int(results.pose_landmarks.landmark[end_idx].x * image.shape[1]),
                                 int(results.pose_landmarks.landmark[end_idx].y * image.shape[0]))
                    cv2.line(image, start_point, end_point, (0, 255, 255), 2)  # Draw connections in yellow

            # Draw a singular dot on the head (nose landmark)
            if 0 < len(results.pose_landmarks.landmark):
                nose_landmark = results.pose_landmarks.landmark[0]
                x = int(nose_landmark.x * image.shape[1])
                y = int(nose_landmark.y * image.shape[0])
                cv2.circle(image, (x, y), 5, (0, 0, 255), -1)  # Draw head point in red

        # Write the processed frame to the output video
        out.write(image)

        frame_count += 1
        print(f"Processed frame: {frame_count}")

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Video processing complete. Total frames processed: {frame_count}. Output saved at: {output_video_path}")