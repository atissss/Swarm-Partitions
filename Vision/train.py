import torch
from ultralytics import YOLO

if __name__ == '__main__':
    
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Training on GPU: {torch.cuda.get_device_name(0)}")

    # model = YOLO("yolov8n.pt")

    # model.train(
    #     data="RoadVehiclesYOLODatasetPro/data.yaml", 
    #     epochs=50, 
    #     save=True,
    #     device=0,
    #     batch=16,
    #     imgsz=640,
    #     workers=2,
    #     patience=10
    # )
    
    model = YOLO("runs/detect/train-6/weights/last.pt")
    model.train(resume = True)