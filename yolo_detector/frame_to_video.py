import cv2
import os

frame_dir = 'output/rgb_frames'
output_video = 'video/output.avi'
fps = 30  # Set to your camera's frame rate

images = sorted([img for img in os.listdir(frame_dir) if img.endswith(".png")])
if not images:
    raise ValueError("No images found in directory.")

frame = cv2.imread(os.path.join(frame_dir, images[0]))
height, width, _ = frame.shape

fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

for image in images:
    video.write(cv2.imread(os.path.join(frame_dir, image)))

video.release()
print("Video saved:", output_video)
