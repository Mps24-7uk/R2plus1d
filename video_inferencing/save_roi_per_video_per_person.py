import argparse
import os
import cv2
import torch
import numpy as np
from time import time
from glob import glob
from ultralytics import YOLO


def process_videos(video_folder, fps_divisor, roi_dir, model_path, device, conf):
    model = YOLO(model_path)

    for video_path in glob(os.path.join(video_folder, "*")):
        video_name = os.path.splitext(os.path.basename(video_path))[0]   # extract video name
        cap = cv2.VideoCapture(video_path)
        fps = round(cap.get(cv2.CAP_PROP_FPS))
        frame_interval = max(1, int(fps / fps_divisor))
        count = 0

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            h, w, _ = frame.shape

            if count % frame_interval == 0:
                results = model.track(frame, persist=True, conf=conf, classes=[0], device=device)

                for box in results[0].boxes:
                    if not torch.is_tensor(box.id):
                        continue

                    x1, y1, x2, y2 = map(round, box.xyxy[0].tolist())
                    tid = int(box.id.item())  # track ID for each person

                    # add 10% padding
                    bw = x2 - x1
                    bh = y2 - y1
                    pad_w = int(0.1 * bw)
                    pad_h = int(0.1 * bh)

                    x1 = max(0, x1 - pad_w)
                    y1 = max(0, y1 - pad_h)
                    x2 = min(w, x2 + pad_w)
                    y2 = min(h, y2 + pad_h)

                    # person folder inside video folder
                    dir_path = os.path.join(roi_dir, video_name, f"person{tid}")
                    os.makedirs(dir_path, exist_ok=True)

                    # crop ROI with padding
                    roi = frame[y1:y2, x1:x2]

                    timestamp = int(time() * 1e9)
                    frame_name = f"{timestamp}_{x1}_{x2}_{y1}_{y2}.jpg"
                    save_path = os.path.join(dir_path, frame_name)
                    cv2.imwrite(save_path, roi)

            count += 1
        cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO video ROI extraction pipeline")
    parser.add_argument("--video_folder", type=str, required=True,
                        help="Path to the input video folder")
    parser.add_argument("--fps_divisor", type=int, default=3,
                        help="Divide FPS by this factor to set frame sampling rate")
    parser.add_argument("--roi_dir", type=str, default="roi",
                        help="Directory to save extracted ROIs")
    parser.add_argument("--model_path", type=str, default="yolopose.pt",
                        help="Path to YOLO model weights (.pt file)")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Computation device: 'cpu', 'cuda', 'cuda:0', etc.")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold for YOLO detection (0.0–1.0)")

    args = parser.parse_args()
    process_videos(
        args.video_folder,
        args.fps_divisor,
        args.roi_dir,
        args.model_path,
        args.device,
        args.conf
    )
