from rosbags.highlevel import AnyReader
from rosbags.image import message_to_cvimage
import cv2
import os
from pathlib import Path


bag_path = '/home/shubh/Desktop/multi-objects.bag'

output_dir = 'output/rgb_frames'
image_topic = '/camera/color/image_raw'  # Adjust as per your bag file

os.makedirs(output_dir, exist_ok=True)

with AnyReader([Path(bag_path)]) as reader:
    for connection, timestamp, rawdata in reader.messages():
        if connection.topic == image_topic:
            msg = reader.deserialize(rawdata, connection.msgtype)
            img = message_to_cvimage(msg, 'bgr8')  # Convert to OpenCV image
            filename = os.path.join(output_dir, f'{timestamp}.png')
            cv2.imwrite(filename, img)
            print(f"Saved {filename}")
