import cv2
import numpy as np
import os
import time
from mss import mss
from datetime import datetime

# --- CONFIGURATION ---
SAVE_PATH = "datasets/ow2_data/train/images"
# Define the capture region (Center of screen 640x640)
# Adjust 'top' and 'left' based on your monitor resolution
monitor = {"top": 220, "left": 640, "width": 640, "height": 640}
CAPTURE_DELAY = 0.5  # Seconds to wait between captures if in 'auto' mode

# Ensure directory exists
os.makedirs(SAVE_PATH, exist_ok=True)


def capture_frames():
    with mss() as sct:
        print("--- OW2 Data Capture Tool Started ---")
        print(f"Saving to: {SAVE_PATH}")
        print("Controls: \n [K] - Save Frame \n [T] - Toggle Auto-Capture \n [Q] - Quit")

        auto_mode = False
        count = 0

        while True:
            # 1. Grab the screen
            img = np.array(sct.grab(monitor))

            # 2. Convert BGRA to BGR for OpenCV
            frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            # 3. Show the "Viewfinder" so you know what the AI will see
            display_frame = frame.copy()
            if auto_mode:
                cv2.putText(display_frame, "AUTO CAPTURE ON", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("Capture Viewfinder", display_frame)

            # 4. Handle Input
            key = cv2.waitKey(1) & 0xFF

            # Manual Save (Press K) or Auto Save
            if key == ord('k') or (auto_mode and (time.time() % CAPTURE_DELAY < 0.05)):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"{SAVE_PATH}/ow2_frame_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                count += 1
                print(f"Captured {count}: {filename}")
                if auto_mode: time.sleep(0.1)  # Prevent double-capturing

            elif key == ord('t'):
                auto_mode = not auto_mode
                print(f"Auto-Capture: {auto_mode}")

            elif key == ord('q'):
                break

    cv2.destroyAllWindows()
    print(f"Session finished. Total images captured: {count}")


if __name__ == "__main__":
    capture_frames()