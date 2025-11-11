# tools/prune.py
import os
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader

# --- your code ---
from model import R2Plus1DModel                     # uses r2plus1d_18 head swap for num_classes=2
# Optional: only needed if you pass --finetune
from dataloader import create_train_val_loaders      # builds (train,val) for (C,T,H,W) clips
# -----------------

def count_params(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    nonzero = 0
    for p in model.parameters():
        if p.is_floating_point():
            nonzero += p.count_nonzero().item()
        else:
            nonzero += p.numel()
    sparsity = 1.0 - (nonzero / max(1, total))
    return total, nonzero, sparsity

def apply_global_l1_prune(model: nn.Module, amount: float):
    """
    Globally prune the smallest-magnitude weights across all Conv3d/Linear layers.
    'amount' is the fraction to prune, e.g., 0.3 = 30%.
    """
    params_to_prune = []
    for m in model.modules():
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            params_to_prune.append((m, "weight"))

    prune.global_unstructured(
        params_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )

    # Make pruning permanent (remove reparam hooks) so state_dict is clean & smaller.
    for (m, _) in params_to_prune:
        try:
            prune.remove(m, "weight")
        except Exception:
            pass

def short_finetune(model, data_root, device, batch=8, epochs=2, lr=1e-4,
                   num_frames=12, image_size=(224, 224), num_workers=2):
    """
    Very short fine-tune to recover accuracy after pruning.
    Uses your existing dataloader API.
    """
    train_loader, _ = create_train_val_loaders(
        root_dir=data_root,
        train_ratio=0.8,
        batch_size=batch,
        num_workers=num_workers,
        num_frames=num_frames,
        image_size=image_size,
        transform=None,          # it already sets default Normalize/Resize internally if None
        shuffle_train=True
    )

    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    for ep in range(1, epochs + 1):
        running = 0.0
        for clips, labels in train_loader:
            clips  = clips.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(clips)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running += loss.item()
        print(f"[finetune] epoch {ep}/{epochs}  loss: {running/len(train_loader):.4f}")

def main():
    ap = argparse.ArgumentParser(description="Global L1 pruning for R2Plus1D.")
    ap.add_argument("--ckpt", required=True, help="Path to trained checkpoint (.pth)")
    ap.add_argument("--out",  default=None, help="Output pruned checkpoint path")
    ap.add_argument("--prune", type=float, default=0.30, help="Fraction to prune [0..1]")
    ap.add_argument("--num_classes", type=int, default=2)
    ap.add_argument("--use_pretrained_backbone", action="store_true",
                    help="Build model with torchvision pretrained backbone before loading your weights")
    ap.add_argument("--finetune", action="store_true", help="Run short fine-tune after pruning")
    ap.add_argument("--data_root", type=str, default=None, help="Dataset root (required for --finetune)")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch",  type=int, default=8)
    ap.add_argument("--num_frames", type=int, default=12)
    ap.add_argument("--image_size", type=int, nargs=2, default=(224, 224))
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    # --- Build model & load weights (matches your train.py/model.py) ---
    model = R2Plus1DModel(num_classes=args.num_classes,
                          pretrained=args.use_pretrained_backbone).to(device)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()

    total0, nonzero0, sp0 = count_params(model)
    print(f"[before] params={total0/1e6:.2f}M  nonzero={nonzero0/1e6:.2f}M  sparsity={sp0*100:.2f}%")

    # --- Prune ---
    apply_global_l1_prune(model, amount=args.prune)

    total1, nonzero1, sp1 = count_params(model)
    print(f"[after ] params={total1/1e6:.2f}M  nonzero={nonzero1/1e6:.2f}M  sparsity={sp1*100:.2f}%")

    # --- Optional short fine-tune to recover accuracy ---
    if args.finetune:
        if not args.data_root:
            raise SystemExit("--finetune requires --data_root")
        short_finetune(model, data_root=args.data_root, device=device,
                       batch=args.batch, epochs=args.epochs, lr=1e-4,
                       num_frames=args.num_frames, image_size=tuple(args.image_size),
                       num_workers=args.num_workers)

    # --- Save clean, pruned checkpoint ---
    out = args.out or (Path(args.ckpt).with_suffix("").as_posix()
                       + f"_pruned{int(args.prune*100)}.pth")
    to_save = {"model_state_dict": model.state_dict()}
    torch.save(to_save, out)
    size_mb = os.path.getsize(out) / (1024 * 1024)
    print(f"[save ] {out} ({size_mb:.2f} MB)")

if __name__ == "__main__":
    main()
