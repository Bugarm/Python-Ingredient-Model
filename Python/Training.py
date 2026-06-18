from py_compile import main

from ultralytics import YOLO
import torch
import os

# Disable CUDA pin memory and enable blocking for Windows stability
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

#print(torch.cuda.is_available())

def main():
    model = YOLO("yolo26m.pt")  # build a new model from YAML

    # Train the model
    results = model.train(
        data="RecipeData\data.yaml", 
        epochs=50, 
        imgsz=640,
        device=[0],
        workers=8,  
        resume=True,
        cache = 'disk',  # Cache images for faster training
        patience = 5,
        classes=[3,7,8,11,14]
    )

if __name__ == '__main__':
    main()