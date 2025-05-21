import cv2
import numpy as np

input_path = "Left_10s.mp4"
output_path = "Left_Cropped_Fullsize.mp4"

# Open video
cap = cv2.VideoCapture(input_path)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define crop region (only crop width)
x_crop = 0
crop_width = 426

# Output writer (same full size as original)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Make sure original frame size is 1080p
    if frame.shape[0] != h or frame.shape[1] != w:
        frame = cv2.resize(frame, (w, h))

    # Start with black frame
    full_frame = np.zeros_like(frame)

    # Copy only cropped section
    full_frame[:, x_crop:x_crop+crop_width] = frame[:, x_crop:x_crop+crop_width]
    out.write(full_frame)

cap.release()
out.release()
print("✅ Saved corrected crop with original frame size:", output_path)