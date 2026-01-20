import cv2
import numpy as np
from mss import mss
from ultralytics import YOLO
import torch_directml

# 1. Load your trained model
# Replace 'best.pt' with the path to your trained weights
model = YOLO("runs/detect/train/weights/best.pt")

# 2. Set up AMD Inference
device = "dml" 

# 3. Define Capture Area (e.g., center of screen 640x640)
# This reduces the amount of pixels the AI has to process, boosting FPS
monitor = {"top": 220, "left": 640, "width": 640, "height": 640}

with mss() as sct:
    while True:
        # Grab screen frame
        img = np.array(sct.grab(monitor))
        
        # Convert BGRA to BGR (OpenCV format)
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # Run YOLO Inference
        results = model.predict(source=frame, device=device, conf=0.5, verbose=False)

        # Process Results
        for r in results:
            annotated_frame = r.plot() # Draws boxes and labels (Genji, Mercy, etc.)
            
            # Extract info (Example: print if a Widowmaker is detected)
            for box in r.boxes:
                class_id = int(box.cls[0])
                label = model.names[class_id]
                if label == "Widowmaker":
                    print("!!! WIDOWMAKER DETECTED - STAY IN COVER !!!")

        # Display (Optional - consumes some FPS)
        cv2.imshow("OW2 AI Detection", annotated_frame)

        if cv2.waitkey(1) & 0xFF == ord("q"):
            break

cv2.destroyAllWindows()