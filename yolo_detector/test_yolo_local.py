
from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")  # or yolov8s.pt or any other variant

# Load your image
image_path = "testimage.jpg"  # Or full path like "/home/yourname/path/test.jpg"
img = cv2.imread(image_path)

# Run detection
results = model(img)

# Show result
annotated = results[0].plot()
cv2.imshow("YOLOv8 Detection", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()

