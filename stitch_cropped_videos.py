import cv2
import numpy as np

left_cap = cv2.VideoCapture("Left_Cropped_Fullsize.mp4")
right_cap = cv2.VideoCapture("Right_10s.mp4")

# Crop bounds for real content in left video
x_crop = 0
crop_width = 426

# Get frame dimensions
h_left = int(left_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = left_cap.get(cv2.CAP_PROP_FPS)
w_right = int(right_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h_right = int(right_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Resize right video to match left height
scale = h_left / h_right
resized_w_right = int(w_right * scale)

# Final stitched width = left crop + resized right width
out = cv2.VideoWriter("Stitched_Final_Clean.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (crop_width + resized_w_right, h_left))

while True:
    ret1, frame1 = left_cap.read()
    ret2, frame2 = right_cap.read()
    if not (ret1 and ret2):
        break

    # Crop out black section from left
    left_cropped = frame1[:, x_crop:x_crop + crop_width]

    # Resize right to match height
    frame2_resized = cv2.resize(frame2, (resized_w_right, h_left))

    # Stitch cleanly
    stitched = np.hstack((left_cropped, frame2_resized))
    out.write(stitched)

left_cap.release()
right_cap.release()
out.release()

print("✅ Clean stitched video saved as: Stitched_Final_Clean.mp4")