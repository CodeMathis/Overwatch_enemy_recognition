import cv2
import numpy as np
import torch_directml
from mss import mss
from ultralytics import YOLO

# 1. Load your trained model
try:
    model = YOLO("./runs/detect/train/weights/best.onnx", task="detect")
except Exception as e:
    print("No model found", e)
    exit()

# 2. Set up Device
device = "cpu" # use this commented line when amd drivers are fixed "dml"

# 3. Capture Area
with mss() as sct:
    # Use primary monitor
    monitor = sct.monitors[1]

    print("--- Overwatch 2 Detection Active ---")
    print("Press 'k' in the window to quit.")

    while True:
        # Grab screen frame
        img = np.array(sct.grab(monitor))

        # Convert BGRA to BGR
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # Run YOLO Inference (optimized resolution)
        results = model.predict(source=frame, device=device, imgsz=640, conf=0.5, verbose=False)

        # Draw results
        annotated_frame = frame.copy()
        for r in results:
            annotated_frame = r.plot()

            # Print detected hero names to console
            for box in r.boxes:
                label = model.names[int(box.cls[0])]
                print(f"Target Found: {label}")

        # Display the live feed
        cv2.imshow("OW2 AI Detection", annotated_frame)

        # FIX: Changed 'waitkey' to 'waitKey' (Capital K)
        if cv2.waitKey(1) & 0xFF == ord("k"):
            break

cv2.destroyAllWindows()