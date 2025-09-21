import argparse
import os
import cv2
import torch
import shutil
import numpy as np
from time import time
from glob import glob
from ultralytics import YOLO
from inference_onnx import prediction


def process_videos(video_folder, fps_divisor, num_frames, roi_dir, cleanup_dir, model_path, predict_dir, device, conf, skip_cleanup):
    model = YOLO(model_path)

    for video_path in glob(os.path.join(video_folder, "*")):
        cap = cv2.VideoCapture(video_path)
        fps = round(cap.get(cv2.CAP_PROP_FPS))
        frame_interval = max(1, int(fps / fps_divisor))
        count = 0

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            if count % frame_interval == 0:
                results = model.track(frame, persist=True, conf=conf, classes=[0], device=device)

                for box in results[0].boxes:
                    if not torch.is_tensor(box.id):
                        continue

                    x1, y1, x2, y2 = map(round, box.xyxy[0].tolist())
                    tid = int(box.id.item())

                    mask = np.zeros(frame.shape, dtype=np.uint8)
                    mask[y1:y2, x1:x2] = frame[y1:y2, x1:x2]

                    dir_path = os.path.join(roi_dir, str(tid))
                    os.makedirs(dir_path, exist_ok=True)

                    timestamp = int(time() * 1e9)
                    frame_name = f"{timestamp}_{x1}_{x2}_{y1}_{y2}.jpg"
                    save_path = os.path.join(dir_path, frame_name)
                    cv2.imwrite(save_path, mask)

                    # Run prediction when enough frames are collected
                    if len(glob(os.path.join(dir_path, "*.jpg"))) == num_frames:
                        label = prediction(dir_path, device, num_frames, batch_size=1)
                        prediction_path = os.path.join(predict_dir, label)
                        os.makedirs(prediction_path, exist_ok=True)

                        shutil.move(
                            dir_path,
                            os.path.join(
                                prediction_path,
                                f"{os.path.basename(video_path)}_@_{timestamp}"
                            )
                        )
            count += 1
        cap.release()

    # Clean up ROI folder if exists and not skipped
    if not skip_cleanup and os.path.exists(cleanup_dir):
        shutil.rmtree(cleanup_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO video fall detection pipeline")
    parser.add_argument("--video_folder", type=str, required=True,
                        help="Path to the input video folder")
    parser.add_argument("--fps_divisor", type=int, default=3,
                        help="Divide FPS by this factor to set frame sampling rate")
    parser.add_argument("--num_frames", type=int, default=9,
                        help="Number of frames to collect before prediction")
    parser.add_argument("--roi_dir", type=str, default="roi",
                        help="Directory to save extracted ROIs")
    parser.add_argument("--cleanup_dir", type=str,
                        default="D:/SEMPRO_AI/fall_detection/v4/R2plus1d/video_inference/roi/",
                        help="Directory to clean up after processing")
    parser.add_argument("--model_path", type=str, default="yolopose.pt",
                        help="Path to YOLO model weights (.pt file)")
    parser.add_argument("--predict_dir", type=str, default="predict",
                        help="Directory to save prediction results")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Computation device: 'cpu', 'cuda', 'cuda:0', etc.")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold for YOLO detection (0.0–1.0)")
    parser.add_argument("--skip_cleanup", action="store_true",
                        help="If set, ROI folders will NOT be deleted after processing")

    args = parser.parse_args()
    process_videos(
        args.video_folder,
        args.fps_divisor,
        args.num_frames,
        args.roi_dir,
        args.cleanup_dir,
        args.model_path,
        args.predict_dir,
        args.device,
        args.conf,
        args.skip_cleanup
    )


#python run.py --video_folder "./video/test_5"  --fps_divisor 3 --num_frames 9
