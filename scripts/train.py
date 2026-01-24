import torch
from ultralytics import YOLO

def start_training():
    # 1. Vérification NVIDIA CUDA
    if torch.cuda.is_available():
        device = 0
        print(f"--- Training on NVIDIA GPU: {torch.cuda.get_device_name(0)} ---")
    else:
        device = "cpu"
        print("--- /!\ CUDA non détecté, utilisation du CPU ---")

    # 2. Charger le modèle (YOLO11 est la version actuelle la plus rapide)
    model = YOLO("yolo26n.pt")

    # 3. Configuration de l'entraînement
    model.train(
        data="./datasets/ow2_data/data.yaml",
        epochs=200,
        imgsz=640,
        batch=12,        # Ajuste à 8 si tu as une erreur de mémoire (VRAM)
        device=device,
        workers=4,
        exist_ok=True,
        amp=True          # Active les Tensor Cores de RTX 3050
    )

    # 4. Export optimal pour NVIDIA
    print("--- Exportation en format TensorRT (.engine) ---")
    model.export(format="onnx", half=True, device=device)

if __name__ == "__main__":
    start_training()