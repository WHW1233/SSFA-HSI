import sys
import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Add local path to import registries
sys.path.append("/home/nercms/compression/qarv_0801")

from lvae.models.registry import get_model
from lvae.datasets.hsi import get_hsi_dateset

# ==========================================
# 1. Downstream Task Network: Lightweight U-Net
# ==========================================
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
        
        # Decoder
        x = self.up1(x3)
        # Handle spatial dimension padding mismatches if any
        diffY = x2.size()[2] - x.size()[2]
        diffX = x2.size()[3] - x.size()[3]
        x = nn.functional.pad(x, [diffX // 2, diffX - diffX // 2,
                                  diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x], dim=1)
        x = self.conv_up1(x)
        
        x = self.up2(x)
        diffY = x1.size()[2] - x.size()[2]
        diffX = x1.size()[3] - x.size()[3]
        x = nn.functional.pad(x, [diffX // 2, diffX - diffX // 2,
                                  diffY // 2, diffY - diffY // 2])
        x = torch.cat([x1, x], dim=1)
        x = self.conv_up2(x)
        
        logits = self.outc(x)
        return logits

# ==========================================
# 2. Joint Loss Function: MSE + SAM (Spectral Angle Mapper)
# ==========================================
def spectral_angle_mapper_loss(x, y, eps=1e-8):
    """
    Computes numerical stable Spectral Angle Mapper (SAM) loss in radians.
    x and y shape: (B, C, H, W)
    """
    # Dot product along channel dimension
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
    epochs=1, 
    batch_size=4, 
    lambda_h=100.0, 
    lambda_m=10.0, 
    alpha=0.1, 
    lr=1e-4, 
    dry_run_steps=3
):
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing co-adaptation training on: {device}")

    # A. Load pre-trained HSI base codec with adapters enabled
    print("Loading base codec 'qarv_hsi_lower' with SSFMA adapters...")
    model = get_model("qarv_hsi_lower", adapted=True)
    model.to(device)
    model.train()

    # B. Load downstream semantic segmenter
    print("Initializing segmentation task U-Net model...")
    num_classes = 10
    segmenter = UNet(in_channels=30, num_classes=num_classes)
    segmenter.to(device)
    segmenter.train()

    # C. Initialize co-adaptation optimizer
    # Optimizes ONLY the adapters and the U-Net parameters
    optimizer = optim.Adam(
        list(model.adapters.parameters()) + list(segmenter.parameters()),
        lr=lr
    )
    
    criterion_task = nn.CrossEntropyLoss()

    # D. Setup HSI DataLoaders (HySpecNet-11k external 8-bit dataset)
    print("Preparing data loaders for HySpecNet-11k external 8-bit dataset...")
    # 'hysp11k-ext-train' points to /media/nercms/EXTERNAL_USB/satellite_data/HSI_data/hyspecnet11k/data
    train_dataset = get_hsi_dateset('hysp11k-ext-train')
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2, 
        drop_last=True
    )

    print(f"Total training patches: {len(train_dataset)}")
    
    # E. Unified Training Loop
    print("\nStarting co-adaptation training pipeline...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_rate = 0.0
        epoch_mse = 0.0
        epoch_sam = 0.0
        epoch_task = 0.0

        for step, batch in enumerate(train_loader):
            # Input batch shape is (B, 30, 128, 128), values in [0.0, 1.0]
            im = batch.to(device)
            B, C, H, W = im.shape

            # Zero gradients
            optimizer.zero_grad()

            # 1. Forward through frozen base VAE with active SSFMA adapters
            # Sample variable rates (lmb) for robustness, or use fixed
            lmb = torch.full((B,), 32.0, device=device)
            x_hat, stats_all = model.forward_end2end(im, lmb)

            # 2. Get reconstructed output
            im_hat = torch.clamp(x_hat, min=-1.0, max=1.0) * 0.5 + 0.5

            # 3. Predict land-cover classes on reconstructed HSI
            pred_mask = segmenter(im_hat)

            # 4. Multi-Task Optimization Loss (RDT Loss)
            # A. Rate (KL Divergence in bpp)
            kl_divergences = [stat['kl'].sum(dim=(1, 2, 3)) for stat in stats_all]
            rate = sum(kl_divergences) / (C * H * W)
            rate = rate.mean(0) * model.log2_e

            # B. Human Distortion (MSE + SAM)
            x_target = model.preprocess_target(im)
            mse_loss_val = nn.MSELoss()(x_hat, x_target)
            sam_loss_val = spectral_angle_mapper_loss(im_hat, im)
            d_human = (1 - alpha) * mse_loss_val + alpha * sam_loss_val

            # C. Task Loss (CrossEntropy)
            # For HyspecNet-11k dry-run, we simulate ground truth masks on the fly
            # In real-world, target_mask is loaded alongside the patch
            target_mask = torch.randint(0, num_classes, (B, H, W), dtype=torch.long, device=device)
            task_loss_val = criterion_task(pred_mask, target_mask)

            # D. Combined Rate-Distortion-Task (RDT) Loss
            total_loss = rate + lambda_h * d_human + lambda_m * task_loss_val

            # Backpropagation (Backprops to both model.adapters and segmenter)
            total_loss.backward()
            optimizer.step()

            # Accumulate logs
            epoch_loss += total_loss.item()
            epoch_rate += rate.item()
            epoch_mse += mse_loss_val.item()
            epoch_sam += sam_loss_val.item()
            epoch_task += task_loss_val.item()

            if (step + 1) % 1 == 0:
                print(
                    f"Epoch [{epoch+1}/{epochs}], Step [{step+1}/{len(train_loader)}], "
                    f"Loss: {total_loss.item():.4f} (Rate: {rate.item():.4f}, "
                    f"MSE: {mse_loss_val.item():.4f}, SAM: {sam_loss_val.item():.4f}, Task: {task_loss_val.item():.4f})"
                )

            # Safety break for dry-run validation on CPU
            if dry_run_steps is not None and (step + 1) >= dry_run_steps:
                print(f"Dry-run limit of {dry_run_steps} steps reached. Stopping step loop.")
                break
        
        avg_loss = epoch_loss / min(len(train_loader), dry_run_steps)
        print(f"\n--- Epoch {epoch+1} Completed. Avg RDT Loss: {avg_loss:.4f} ---")

    # Save trained adapters and segmenter state dicts
    save_path = "/home/nercms/compression/qarv_0801/adapters_and_task_head.pt"
    torch.save({
        'adapters': model.adapters.state_dict(),
        'segmenter': segmenter.state_dict(),
    }, save_path)
    print(f"Saved co-adapted weights successfully to: {save_path}")
    print("Co-adaptation integration pipeline run successfully completed!")

if __name__ == "__main__":
    # Default to a 3-step CPU dry-run verification
    run_co_adaptation(epochs=1, dry_run_steps=3)
