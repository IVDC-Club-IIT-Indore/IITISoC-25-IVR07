from ultralytics import YOLO

# Load the YOLOv8n model (downloads weights automatically if not present)
model = YOLO("yolov8n.pt")

# Path to your extracted video
video_path = "video/output.avi"


# Run object detection on the video and save the results
results = model.predict(
    source=video_path,    # Path to video
    save=True,            # Save the output video with detections
    show=True,            # (Optional) Show results in a window
    conf=0.25,            # Confidence threshold (adjust as needed)
    imgsz=640             # Image size for inference (default 640)
)

print("Detection complete. Annotated video saved to the 'runs/predict' directory.")
