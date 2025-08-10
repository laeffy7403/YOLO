#!/usr/bin/env python3

from ultralytics import YOLO
import cv2
import time
from collections import Counter
from pathlib import Path
import mss
import numpy as np

# === CONFIG ===
MODEL_PATH = "runs/detect/custom/weights/best.pt"
VIDEO_PATH = "shiba1.mp4"
OUTPUT_VIDEO = "output_annotated.mp4"

def count_summary(results_list):
    class_counter = Counter()
    for result in results_list:
        for c in result.boxes.cls.tolist():
            class_counter[int(c)] += 1

    print("\n📊 [TRACKING SUMMARY]")
    for class_id, count in class_counter.items():
        name = result.names[class_id]
        print(f" - {name}: {count} detected")

    if not class_counter:
        print(" - No objects detected. 😔")


def track_from_video(model):
    cap = cv2.VideoCapture(VIDEO_PATH)
    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = int(cap.get(5))
    out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    results_log = []
    print("[INFO] Tracking from video... Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        results_log.append(results[0])
        annotated = results[0].plot()
        out.write(annotated)
        cv2.imshow("Tracking - Video", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Video tracking done. Saved to {OUTPUT_VIDEO}")
    count_summary(results_log)


def track_from_camera(model):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Failed to access webcam.")
        return

    results_log = []
    print("[INFO] Tracking from webcam... Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        results_log.append(results[0])
        annotated = results[0].plot()
        cv2.imshow("Tracking - Webcam", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Webcam tracking ended.")
    count_summary(results_log)


def track_from_screen(model):
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        print("[INFO] Tracking screen... Press 'q' to quit.")
        while True:
            screenshot = sct.grab(monitor)
            img = np.array(screenshot)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            results = model(img)
            annotated = results[0].plot()

            cv2.imshow("Tracking - Screen", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        print("[INFO] Screen tracking ended.")


def run_tracking_loop(model):
    while True:
        print("\n=== Select Tracking Source ===")
        print("1. Track from video file")
        print("2. Track from built-in webcam")
        print("3. Exit")
        choice = input("Enter choice (1, 2, 3): ")

        if choice == '1':
            track_from_video(model)
        elif choice == '2':
            track_from_camera(model)
        elif choice == '3':
            print("[INFO] Exiting tracking loop.")
            break
        else:
            print("❌ Invalid choice. Try again.")


if __name__ == "__main__":
    print("[INFO] Loading model...")
    model = YOLO(MODEL_PATH)
    run_tracking_loop(model)
