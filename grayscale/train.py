# train.py
import os
import re
import json
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from datetime import datetime

from dataloader import create_train_val_loaders, KINETICS_MEAN, KINETICS_STD
from model import R2Plus1DModel
import torchvision.transforms as T


def create_experiment_folder(root: str = "run", name: str | None = None) -> str:
    """
    Create an experiment directory:
      - If `name` is given, use run/<name> (e.g., run/exp1).
      - Otherwise, auto-increment to run/expN (exp1, exp2, ...).
    Returns the absolute path to the created directory.
    """
    os.makedirs(root, exist_ok=True)

    if name:
        exp_dir = os.path.join(root, name)
        os.makedirs(exp_dir, exist_ok=True)
        return exp_dir

    # auto-increment exp number
    existing = [
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    ]
    nums = []
    for d in existing:
        m = re.fullmatch(r"exp(\d+)", d)
        if m:
            nums.append(int(m.group(1)))
    next_id = (max(nums) + 1) if nums else 1
    exp_dir = os.path.join(root, f"exp{next_id}")
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(loader, desc="Train", leave=False)
    for clips, labels in pbar:
        clips = clips.to(device, non_blocking=True)   # (B, C, T, H, W) with C=1
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(clips)                        # (B, 2)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * clips.size(0)
        _, pred = outputs.max(1)
        correct += pred.eq(labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix(loss=running_loss / max(1, total),
                         acc=correct / max(1, total))

    return running_loss / max(1, total), correct / max(1, total)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(loader, desc="Val  ", leave=False)
    for clips, labels in pbar:
        clips = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(clips)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * clips.size(0)
        _, pred = outputs.max(1)
        correct += pred.eq(labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix(loss=running_loss / max(1, total),
                         acc=correct / max(1, total))

    return running_loss / max(1, total), correct / max(1, total)


def main():
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----- Config -----
    data_root   = "./data"     # expects fall/ and no_fall/ subfolders
    num_epochs  = 10
    batch_size  = 4
    num_workers = 2
    lr          = 1e-4
    num_frames  = 12
    image_size  = (224, 224)   # recommended for R(2+1)D
    run_root    = "run"        # base folder to store experiments
    exp_name    = None         # set to e.g. "exp1" to force a name; else auto-increment

    # ----- Create experiment directory (e.g., ./run/exp1) -----
    save_dir = create_experiment_folder(run_root, exp_name)
    print(f"[INFO] Saving artifacts to: {os.path.abspath(save_dir)}")
    ckpt_path = os.path.join(save_dir, "best_model.pth")
    cfg_path  = os.path.join(save_dir, "config.json")

    # Save (or update) a small config file for traceability
    config = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_root": data_root,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "lr": lr,
        "num_frames": num_frames,
        "image_size": image_size,
        "device": str(device),
        "input_channels": 1,
    }
    try:
        with open(cfg_path, "w") as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        print(f"[WARN] Could not write config.json: {e}")

    # Per-frame transforms (for grayscale: mean/std are length-1)
    transform = T.Compose([
        T.Resize(image_size),
        T.ToTensor(),  # PIL "L" -> (1, H, W)
        T.Normalize(mean=KINETICS_MEAN, std=KINETICS_STD),
    ])

    # ----- Data -----
    train_loader, val_loader = create_train_val_loaders(
        root_dir=data_root,
        train_ratio=0.8,
        batch_size=batch_size,
        num_workers=num_workers,
        num_frames=num_frames,
        image_size=image_size,
        transform=transform,
        shuffle_train=True
    )

    # ----- Model, Loss, Optim -----
    # 🔸 IMPORTANT: in_channels=1 for grayscale
    model = R2Plus1DModel(num_classes=2, pretrained=True, in_channels=1).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # ----- Load previous best (if exists in this exp dir) -----
    best_val_loss = float("inf")
    best_val_acc  = 0.0
    if os.path.exists(ckpt_path):
        try:
            prev = torch.load(ckpt_path, map_location="cpu")
            if "best_val_loss" in prev:
                best_val_loss = float(prev["best_val_loss"])
                best_val_acc  = float(prev.get("best_val_acc", 0.0))
                print(f"[INFO] Loaded previous best from {ckpt_path}: "
                      f"val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.4f}")
            # Optionally warm-start weights:
            if "model_state_dict" in prev:
                model.load_state_dict(prev["model_state_dict"], strict=False)
                print("[INFO] Warm-started model from existing checkpoint.")
        except Exception as e:
            print(f"[WARN] Could not load existing checkpoint: {e}")

    # ----- Train loop -----
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch [{epoch}/{num_epochs}]")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc     = validate(model, val_loader, criterion, device)

        print(f" -> Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f" -> Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        # ----- Save if improved on previous best val loss -----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc  = val_acc
            to_save = {
                "epoch": epoch,
                "best_val_loss": best_val_loss,
                "best_val_acc": best_val_acc,
                "model_state_dict": model.state_dict(),
            }
            torch.save(to_save, ckpt_path)
            print(f"[SAVE] New best model saved to {ckpt_path} "
                  f"(val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.4f})")


if __name__ == "__main__":
    main()
