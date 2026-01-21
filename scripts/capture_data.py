import cv2
import numpy as np
import os
import time
from mss import mss
from datetime import datetime
from pynput import keyboard

# --- CONFIGURATION ---
SAVE_PATH = "./datasets/preprocessed_images"
TARGET_SIZE = (640, 640)
SMART_DELAY = 2  # Check for enemies every 2s
RANDOM_DELAY = 10  # Capture background every 10s
MIN_RED_PIXELS = 500

# Red HSV ranges for Overwatch 2 enemy outlines
LOWER_RED1, UPPER_RED1 = np.array([0, 200, 150]), np.array([5, 255, 255])
LOWER_RED2, UPPER_RED2 = np.array([175, 200, 150]), np.array([180, 255, 255])

os.makedirs(SAVE_PATH, exist_ok=True)

auto_mode = False
quit_program = False


def on_press(key):
    global auto_mode, quit_program
    try:
        if key.char == 'i':
            auto_mode = not auto_mode
            print(f">>> Capture System: {'ON' if auto_mode else 'OFF'}")
        elif key.char == 'k':
            quit_program = True
    except AttributeError:
        pass


def is_interesting(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.bitwise_or(cv2.inRange(hsv, LOWER_RED1, UPPER_RED1),
                          cv2.inRange(hsv, LOWER_RED2, UPPER_RED2))
    return np.sum(mask > 0) > MIN_RED_PIXELS


def capture_frames():
    global quit_program, auto_mode
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    with mss() as sct:
        monitor = sct.monitors[1]
        print("--- OW2 Hybrid Capture Active ---")
        print("[I] Toggle | [K] Quit")

        count_smart = 0
        count_random = 0
        last_smart_time = 0
        last_random_time = time.time()

        while not quit_program:
            if auto_mode:
                current_time = time.time()
                sct_img = sct.grab(monitor)
                frame_full = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)

                # --- LOGIC 1: SMART CAPTURE (Enemies present) ---
                if current_time - last_smart_time >= SMART_DELAY:
                    if is_interesting(frame_full):
                        save_frame(frame_full, "smart")
                        count_smart += 1
                        last_smart_time = current_time
                        print(f"[+] Smart enemy capture saved ({count_smart})")

                # --- LOGIC 2: RANDOM CAPTURE (Background/Negative Data) ---
                if current_time - last_random_time >= RANDOM_DELAY:
                    save_frame(frame_full, "random")
                    count_random += 1
                    last_random_time = current_time
                    print(f"[*] Random background saved ({count_random})")

            time.sleep(0.01)
    listener.stop()


def save_frame(frame, prefix):
    resized = cv2.resize(frame, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    cv2.imwrite(f"{SAVE_PATH}/{prefix}_{ts}.jpg", resized)


if __name__ == "__main__":
    capture_frames()