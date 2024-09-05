"""
Modified Example script from:
https://docs.ultralytics.com/modes/track/#tracking
"""

from collections import defaultdict

import cv2
import numpy as np

from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Open the video file
video_path = "/home/albert/doodad-output/24-01-19-BC-image-PPObjToPotToStove/24-01-19-BC-image-PPObjToPotToStove_2024_01_19_23_37_18_id000--s411/videos_eval/fr1_0/epoch500_7.mp4"
cap = cv2.VideoCapture(video_path)
out_path = "/home/albert/dev/deoxys_v2/deoxys/scripts/20240402.mp4"

# Store the track history
track_history = defaultdict(lambda: [])

video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
FPS = float(np.ceil(video_len/3.0))
imsize, imsize = 128, 128
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
writer = cv2.VideoWriter(out_path, fourcc, FPS, (imsize, imsize))

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, conf=0.1, persist=True)

        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()
        # track_ids = results[0].boxes.id.int().cpu().tolist()

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        writer.write(annotated_frame)
    else:
        # Break the loop if the end of the video is reached
        writer = None
        break

cap.release()
cv2.destroyAllWindows()
