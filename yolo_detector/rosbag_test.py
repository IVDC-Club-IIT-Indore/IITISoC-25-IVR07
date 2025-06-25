from pathlib import Path
from rosbags.highlevel import AnyReader

def list_rosbag_topics(bag_path):
    with AnyReader([Path(bag_path)]) as reader:
        topics = reader.topics
        for topic, info in topics.items():
            print(f"Topic: {topic}")
            print(f"  Type: {info.msgtype}")
            print(f"  Message count: {info.msgcount}")
            print("-" * 40)

# Replace 'your_file.bag' with your actual bag file path
list_rosbag_topics('../home/shubh/Desktop/multi-object.bag') # change this to your bag file path (discord wale link se download kar lena)
