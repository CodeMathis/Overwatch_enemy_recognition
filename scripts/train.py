import torch
import torch_directml
from ultralytics import YOLO
import shutil

def start_training():
    # 1. Initialize AMD GPU via DirectML
    device = torch_directml.device()
    print(f"--- Training on AMD GPU: {torch_directml.device_name(0)} ---")

    # 2. Load the newest YOLO26 Nano (Optimized for speed)
    model = YOLO("yolo26n.pt")

    # 3. Training Configuration
    model.train(
        data="../datasets/ow2_data/data.yaml",
        epochs=100,
        imgsz=640,       # Standard resolution for balanced speed/accuracy
        batch=16,        # Your 7700 XT (12GB) can easily handle 16-32
        device=device,    # DirectML for AMD support
        workers=4,       # Parallel data loading
        exist_ok=True,
        amp=True         # Automatic Mixed Precision for faster AMD training
    )

    # 4. Export for maximum performance
    # Exporting to ONNX allows for faster real-time inference later
    model.export(format="onnx", half=True)

    shutil.move("./yolo26n.pt", "../models/yolo26n.pt")
    print("--- Training Complete! Model saved to ../models/yolo26n.pt ---")

if __name__ == "__main__":
    start_training()