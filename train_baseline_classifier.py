import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.append("/home/nercms/compression/qarv_0801")
from lvae.datasets.hsi import get_hsi_dateset

# ==========================================
# Downstream Task Network: Classifier
# ==========================================
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

def train_baseline(epochs=10, batch_size=32, lr=1e-4, dry_run_steps=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing baseline training on: {device}")

    model = Classifier(in_channels=30, num_classes=9)
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    print("Preparing train and val data loaders for WHU-OHS dataset...")
    train_dataset = get_hsi_dateset('ohs-train')
    val_dataset = get_hsi_dateset('ohs-val')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"Total training patches: {len(train_dataset)}")
    print(f"Total validation patches: {len(val_dataset)}")

    log_file_path = "/home/nercms/compression/qarv_0801/baseline_classification_results.txt"
    with open(log_file_path, "w") as f:
        f.write("Epoch\tTrain_Loss\tVal_Accuracy\n")

    print("\nStarting baseline classification training...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for step, batch in enumerate(train_loader):
            im, target = batch
            im = im.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            pred_logits = model(im)
            loss = criterion(pred_logits, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            
            if (step + 1) % 10 == 0 or (step + 1) == 1:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{step+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

            if dry_run_steps is not None and (step + 1) >= dry_run_steps:
                print(f"Dry-run limit of {dry_run_steps} steps reached. Stopping training step loop.")
                break
                
        avg_train_loss = epoch_loss / min(len(train_loader), dry_run_steps if dry_run_steps else len(train_loader))
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        print(f"Evaluating validation accuracy...")
        with torch.no_grad():
            for step, batch in enumerate(val_loader):
                im, target = batch
                im = im.to(device)
                target = target.to(device)
                
                pred_logits = model(im)
                _, predicted = torch.max(pred_logits.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
                if dry_run_steps is not None and (step + 1) >= dry_run_steps:
                    print(f"Dry-run limit of {dry_run_steps} steps reached. Stopping validation loop.")
                    break

        val_accuracy = 100 * correct / total
        print(f"--- Epoch [{epoch+1}/{epochs}] Completed | Avg Train Loss: {avg_train_loss:.4f} | Val Accuracy: {val_accuracy:.2f}% ---")

        with open(log_file_path, "a") as f:
            f.write(f"{epoch+1}\t{avg_train_loss:.4f}\t{val_accuracy:.2f}\n")

    torch.save(model.state_dict(), "/home/nercms/compression/qarv_0801/baseline_classifier.pt")
    print(f"\nTraining finished! Results logged to {log_file_path}")

if __name__ == "__main__":
    # If running on Orin for testing, set dry_run_steps. On GPU server, remove dry_run_steps.
    train_baseline(epochs=3, batch_size=4, lr=1e-4, dry_run_steps=3)
