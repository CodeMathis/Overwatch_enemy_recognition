import cv2
import numpy as np
import os
import time
from mss import mss
from datetime import datetime
from pynput import keyboard

# --- CONFIGURATION ---
SAVE_PATH = "../datasets/ow2_data/train/images"
TARGET_SIZE = (640, 640)
CAPTURE_DELAY = 0.5

os.makedirs(SAVE_PATH, exist_ok=True)

# Global state variables
auto_mode = False
save_frame_signal = False
quit_program = False


def on_press(key):
    global auto_mode, save_frame_signal, quit_program
    try:
        if key.char == 'k':
            save_frame_signal = True
        elif key.char == 'i':
            auto_mode = not auto_mode
            print(f">>> Auto-Capture: {'ON' if auto_mode else 'OFF'}")
        elif key.char == 'l':
            quit_program = True
    except AttributeError:
        pass  # Ignore special keys like Shift/Alt


def capture_frames():
    global save_frame_signal, quit_program, auto_mode

    # Start the global listener in the background
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    with mss() as sct:
        monitor = sct.monitors[1]
        print("--- OW2 Global Data Capture Started ---")
        print("You can now go back into Overwatch 2.")
        print("Controls: [K] Save, [I] Toggle Auto, [L] Leave")

        count = 0
        last_auto_time = time.time()

        while not quit_program:
            sct_img = sct.grab(monitor)
            frame_full = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)
            frame_resized = cv2.resize(frame_full, TARGET_SIZE, interpolation=cv2.INTER_AREA)

            # Check if it's time to auto-save
            if auto_mode and (time.time() - last_auto_time >= CAPTURE_DELAY):
                save_frame_signal = True
                last_auto_time = time.time()

            # Save Execution
            if save_frame_signal:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"{SAVE_PATH}/ow2_{timestamp}.jpg"
                cv2.imwrite(filename, frame_resized)
                count += 1
                print(f"[{count}] Saved: {filename}")
                save_frame_signal = False  # Reset signal

    listener.stop()
    cv2.destroyAllWindows()
    print(f"\nSession finished. Total images: {count}")


if __name__ == "__main__":
    capture_frames()