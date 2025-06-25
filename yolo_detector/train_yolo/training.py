from ultralytics import YOLO
import torch
import yaml

class COCOBallHumanTrainer:
    def __init__(self, model_size='s', gpu_id=0):
        """
        COCO trainer optimized for ball and human detection
        
        Args:
            model_size: 'n', 's', 'm', 'l', 'x'
            gpu_id: GPU device ID
        """
        self.device = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'
        self.model_size = model_size
        self.model = None
        
        # Target classes from COCO dataset
        self.target_classes = {
            0: 'person',      # Human detection
            32: 'sports ball' # Ball detection
        }
        
        if torch.cuda.is_available():
            print(f"Using GPU: {torch.cuda.get_device_name(gpu_id)}")
            print(f"Available VRAM: {torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3:.1f} GB")
        
    def setup_model(self):
        """Setup YOLOv8/YOLOv11 model with COCO pre-trained weights"""
        model_name = f"yolo11{self.model_size}.pt"  # Using YOLOv11 as shown in search results
        self.model = YOLO(model_name)
        print(f"Loaded pre-trained {model_name} with COCO weights")
    
    def train_on_coco(self, epochs=100):
        """
        Train on full COCO dataset with GPU optimization
        """
        if self.model is None:
            self.setup_model()
        
        # Determine optimal batch size based on GPU memory
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        if gpu_memory_gb >= 24:
            batch_size = 32
        elif gpu_memory_gb >= 16:
            batch_size = 24
        elif gpu_memory_gb >= 12:
            batch_size = 16
        elif gpu_memory_gb >= 8:
            batch_size = 12
        else:
            batch_size = 8
        
        print(f"Training on full COCO dataset with batch size: {batch_size}")
        
        # Training command from search results[1]
        results = self.model.train(
            data="coco.yaml",  # Full COCO dataset
            epochs=epochs,
            batch=batch_size,
            imgsz=640,
            device=self.device,
            # GPU optimizations
            amp=True,           # Mixed precision training
            cache=True,         # Cache images in memory
            workers=8,          # Parallel data loading
            # Performance settings
            patience=50,
            save_period=10,
            plots=True,
            name='coco_ball_human_training'
        )
        
        return results
    
    def train_on_coco8_quick_test(self):
        """
        Quick training test on COCO8 (8 images) for validation
        """
        if self.model is None:
            self.setup_model()
        
        print("Running quick test on COCO8 dataset...")
        
        # Training on COCO8 as shown in search results[5]
        results = self.model.train(
            data="coco8.yaml",  # Small COCO subset for testing
            epochs=10,          # Few epochs for quick test
            batch=4,
            imgsz=640,
            device=self.device,
            name='coco8_quick_test'
        )
        
        return results
    
    def fine_tune_for_ball_human(self, epochs=50):
        """
        Fine-tune model focusing on ball and human detection
        """
        if self.model is None:
            self.setup_model()
        
        print("Fine-tuning for ball and human detection...")
        
        # Fine-tuning with lower learning rate as suggested in search results[2]
        results = self.model.train(
            data="coco.yaml",
            epochs=epochs,
            batch=16,
            imgsz=640,
            device=self.device,
            # Fine-tuning specific settings
            lr0=0.001,          # Lower learning rate for fine-tuning
            warmup_epochs=3,
            # Focus on target classes
            amp=True,
            cache=True,
            name='coco_fine_tuned_ball_human'
        )
        
        return results

# Usage
trainer = COCOBallHumanTrainer(model_size='s', gpu_id=0)

# Option 1: Quick test on COCO8
quick_results = trainer.train_on_coco8_quick_test()

# Option 2: Full COCO training
# full_results = trainer.train_on_coco(epochs=100)

# Option 3: Fine-tuning approach
# fine_tune_results = trainer.fine_tune_for_ball_human(epochs=50)
