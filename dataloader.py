# dataloader.py
import os
from typing import List, Tuple
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T

KINETICS_MEAN = [0.43216, 0.394666, 0.37645]
KINETICS_STD  = [0.22803, 0.22145, 0.216989]

class VideoFolderDataset(Dataset):
    """
    Expects a directory structure:
      root/
        fall/
          video_001/ frame_0001.jpg ... frame_0012.jpg
          video_002/ ...
        no_fall/
          video_101/ ...
    Each video folder is one sample (sequence).
    Returns:
      clip: Tensor (C, T, H, W)
      label: 0 = fall, 1 = no_fall
    """
    def __init__(
        self,
        root_dir: str,
        num_frames: int = 12,
        image_size: Tuple[int, int] = (112, 112),
        transform=None
    ):
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.image_size = image_size

        # Default per-frame transform if none provided
        if transform is None:
            self.transform = T.Compose([
                T.Resize(image_size),             # 112x112 recommended for R(2+1)D
                T.ToTensor(),
                T.Normalize(mean=KINETICS_MEAN, std=KINETICS_STD),
            ])
        else:
            self.transform = transform

        self.samples: List[Tuple[List[str], int]] = []  # [(frame_paths, label)]
        class_dirs = [("fall", 0), ("no_fall", 1)]

        for class_name, label in class_dirs:
            class_path = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_path):
                print(f"[WARN] Missing class folder: {class_path}. Skipping.")
                continue

            # Each subfolder is a video/sequence
            for d in sorted(os.listdir(class_path)):
                vid_dir = os.path.join(class_path, d)
                if not os.path.isdir(vid_dir):
                    continue
                frames = sorted([
                    os.path.join(vid_dir, f)
                    for f in os.listdir(vid_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ])
                if len(frames) == 0:
                    continue

                # Enforce exactly num_frames via reflect-padding/truncation
                frames = self._fix_length_reflect(frames, self.num_frames)
                if len(frames) == self.num_frames:
                    self.samples.append((frames, label))

        if len(self.samples) == 0:
            print(f"[WARN] No samples found under {root_dir}. Check data structure.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        frame_paths, label = self.samples[idx]
        clip = []

        for fp in frame_paths:
            img = Image.open(fp).convert("RGB")
            img = self.transform(img)  # (C, H, W)
            clip.append(img)

        # Stack to (T, C, H, W) then permute to (C, T, H, W)
        clip = torch.stack(clip, dim=0).permute(1, 0, 2, 3).contiguous()
        return clip, torch.tensor(label, dtype=torch.long)

    @staticmethod
    def _fix_length_reflect(frame_files: List[str], target_len: int) -> List[str]:
        """Reflect-pad/truncate a list of frame paths to target_len."""
        n = len(frame_files)
        if n == target_len:
            return frame_files
        if n > target_len:
            return frame_files[:target_len]

        # Reflect pad: e.g., [0,1,2,3] -> extend with [2,1] [2,1] ...
        out = frame_files[:]
        if n == 1:
            # Single frame: just repeat it
            while len(out) < target_len:
                out.append(frame_files[0])
            return out

        mirror_idx = list(range(n - 2, 0, -1))  # exclude endpoints
        while len(out) < target_len:
            for j in mirror_idx:
                out.append(frame_files[j])
                if len(out) >= target_len:
                    break
        return out


def create_train_val_loaders(
    root_dir: str,
    train_ratio: float = 0.8,
    batch_size: int = 4,
    num_workers: int = 0,
    num_frames: int = 12,
    image_size: Tuple[int, int] = (112, 112),
    transform=None,
    shuffle_train: bool = True,
):
    dataset = VideoFolderDataset(
        root_dir=root_dir,
        num_frames=num_frames,
        image_size=image_size,
        transform=transform
    )
    total = len(dataset)
    train_len = int(total * train_ratio)
    val_len = total - train_len
    train_set, val_set = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=shuffle_train,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader
