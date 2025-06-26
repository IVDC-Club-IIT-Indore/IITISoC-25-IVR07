import cv2
from ultralytics import YOLO

class COCOBallHumanDetector:
    def __init__(self, model_path='runs/detect/coco_ball_human_training/weights/best.pt'):
        """Real-time detector for ball and human detection"""
        self.model = YOLO(model_path)
        
        # COCO class IDs for filtering
        self.target_classes = {0: 'person', 32: 'sports ball'}
        self.colors = {
            0: (0, 255, 0),    # Green for person
            32: (0, 0, 255)    # Red for sports ball
        }
    
    def run_webcam_detection(self):
        """Run real-time detection on webcam"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("COCO Ball & Human Detection Active!")
        print("Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run detection
            results = self.model(frame, conf=0.5, device='cuda')
            
            # Filter for target classes only
            annotated_frame = self.draw_filtered_detections(frame, results[0])
            
            cv2.imshow('COCO Ball & Human Detection', annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def draw_filtered_detections(self, frame, results):
        """Draw only ball and human detections"""
        annotated_frame = frame.copy()
        
        boxes = results.boxes
        if boxes is not None:
            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                # Filter for target classes only
                if class_id in self.target_classes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Draw bounding box
                    color = self.colors[class_id]
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Add label
                    label = f"{self.target_classes[class_id]}: {confidence:.2f}"
                    cv2.putText(annotated_frame, label, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return annotated_frame

# Test the detector
# detector = COCOBallHumanDetector()
# detector.run_webcam_detection()
