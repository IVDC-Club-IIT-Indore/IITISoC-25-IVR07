# Start with pre-trained model and fine-tune
import training as training_yolo
trainer = training_yolo.COCOBallHumanTrainer(model_size='s')
fine_tune_results = trainer.fine_tune_for_ball_human(epochs=50)
