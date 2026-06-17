import sys
import os
import math
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Add local path to import registries
sys.path.append("/home/nercms/compression/qarv_0801")

from lvae.models.registry import get_model
from lvae.datasets.hsi import get_hsi_dateset

# ==========================================
# 1. Downstream Task Networks
# ==========================================

# A. Image Classifier
class Classifier(nn.Module):
    def __init__(self, in_channels=30, num_classes=9):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 32x32
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 16x16
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits


# B. Pixel-Level Segmentation UNet
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=30, num_classes=10):
        super().__init__()
        self.inc = DoubleConv(in_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(128, 64)
        
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(64, 32)
        
        self.outc = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        
        x = self.up1(x3)
        diffY = x2.size()[2] - x.size()[2]
        diffX = x2.size()[3] - x.size()[3]
        x = nn.functional.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x], dim=1)
        x = self.conv_up1(x)
        
        x = self.up2(x)
        diffY = x1.size()[2] - x.size()[2]
        diffX = x1.size()[3] - x.size()[3]
        x = nn.functional.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x1, x], dim=1)
        x = self.conv_up2(x)
        
        logits = self.outc(x)
        return logits

# ==========================================
# 2. Joint Loss Function: MSE + SAM
# ==========================================
def spectral_angle_mapper_loss(x, y, eps=1e-8):
    dot = torch.sum(x * y, dim=1)
    norm_x = torch.norm(x, p=2, dim=1)
    norm_y = torch.norm(y, p=2, dim=1)
    cos_theta = dot / (norm_x * norm_y + eps)
    cos_theta = torch.clamp(cos_theta, -1.0 + eps, 1.0 - eps)
    sam = torch.acos(cos_theta)
    return sam.mean()


# ==========================================
# 3. Main Co-Adaptation Optimization Loop
# ==========================================
def run_co_adaptation(
    task_type="classification",
    dataset_name="ohs-train",
    epochs=1, 
    batch_size=4, 
    lambda_h=100.0, 
    lambda_m=10.0, 
    alpha=0.1, 
    lr=1e-4, 
    dry_run_steps=3
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing co-adaptation training on: {device} | Task: {task_type}")

    print("Loading base codec 'qarv_hsi_lower' with SSFMA adapters...")
    model = get_model("qarv_hsi_lower", pretrained="/media/nercms/EXTERNAL_USB/orin_backup/projects/runs/qarv/qarv_hsi_lower_0/best.pt", adapted=True)
    model.to(device)
    model.train()

    print(f"Initializing {task_type} task model...")
    if task_type == "classification":
        num_classes = 9
        task_model = Classifier(in_channels=30, num_classes=num_classes)
    else:
        num_classes = 10 # Pavia classes: 0 to 9
        task_model = UNet(in_channels=30, num_classes=num_classes)
        
    task_model.to(device)
    task_model.train()

    optimizer = optim.Adam(
        list(model.adapters.parameters()) + list(task_model.parameters()),
        lr=lr
    )
    
    # We ignore class 0 for pavia dataset pixel-level classification (usually 0 is unlabeled/background)
    if task_type == "segmentation" and "pavia" in dataset_name:
        criterion_task = nn.CrossEntropyLoss(ignore_index=0)
    else:
        criterion_task = nn.CrossEntropyLoss()

    print(f"Preparing data loaders for {dataset_name} dataset...")
    train_dataset = get_hsi_dateset(dataset_name)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2, 
        drop_last=True
    )

    print(f"Total training patches: {len(train_dataset)}")
    
    print("\nStarting co-adaptation training pipeline...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for step, batch in enumerate(train_loader):
            im, target = batch
            im = im.to(device)
            target = target.to(device)
            B, C, H, W = im.shape

            optimizer.zero_grad()

            lmb = torch.full((B,), 32.0, device=device)
            x_hat, stats_all = model.forward_end2end(im, lmb)

            im_hat = torch.clamp(x_hat, min=-1.0, max=1.0) * 0.5 + 0.5

            pred_logits = task_model(im_hat)

            # RDT Loss
            kl_divergences = [stat['kl'].sum(dim=(1, 2, 3)) for stat in stats_all]
            rate = sum(kl_divergences) / (C * H * W)
            rate = rate.mean(0) * model.log2_e

            x_target = model.preprocess_target(im)
            mse_loss_val = nn.MSELoss()(x_hat, x_target)
            sam_loss_val = spectral_angle_mapper_loss(im_hat, im)
            d_human = (1 - alpha) * mse_loss_val + alpha * sam_loss_val

            task_loss_val = criterion_task(pred_logits, target)

            total_loss = rate + lambda_h * d_human + lambda_m * task_loss_val

            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()

            if (step + 1) % 1 == 0:
                print(
                    f"Epoch [{epoch+1}/{epochs}], Step [{step+1}/{len(train_loader)}], "
                    f"Loss: {total_loss.item():.4f} (Rate: {rate.item():.4f}, "
                    f"MSE: {mse_loss_val.item():.4f}, SAM: {sam_loss_val.item():.4f}, Task: {task_loss_val.item():.4f})"
                )

            if dry_run_steps is not None and (step + 1) >= dry_run_steps:
                print(f"Dry-run limit of {dry_run_steps} steps reached. Stopping step loop.")
                break
        
        avg_loss = epoch_loss / min(len(train_loader), dry_run_steps if dry_run_steps else len(train_loader))
        print(f"\n--- Epoch {epoch+1} Completed. Avg RDT Loss: {avg_loss:.4f} ---")

    save_path = f"/home/nercms/compression/qarv_0801/adapters_and_{task_type}_head.pt"
    torch.save({
        'adapters': model.adapters.state_dict(),
        f'{task_type}': task_model.state_dict(),
    }, save_path)
    print(f"Saved co-adapted weights successfully to: {save_path}")
    print("Co-adaptation integration pipeline run successfully completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Co-adaptation Training")
    parser.add_argument("--task_type", type=str, default="classification", choices=["classification", "segmentation"])
    parser.add_argument("--dataset", type=str, default="ohs-train")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--dry_run_steps", type=int, default=None)
    args = parser.parse_args()

    run_co_adaptation(
        task_type=args.task_type,
        dataset_name=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        dry_run_steps=args.dry_run_steps
    )
