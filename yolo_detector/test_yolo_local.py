
from ultralytics import YOLO
import cv2

# Load model
model = YOLO("yolov8n.pt")

# Load image (replace with your test image path)
img = cv2.imread("test.jpg")
results = model(img, show=True)

# Wait to close
cv2.waitKey(0)
cv2.destroyAllWindows()
