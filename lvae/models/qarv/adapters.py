import torch
import torch.nn as nn

class SpMA(nn.Module):
    """
    Spectral Modulation Adapter (SpMA):
    Learns band-wise/channel-wise semantic attention to select diagnostic spectral features.
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        reduction = max(1, reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.fc(x)
        return x * w

class SMA(nn.Module):
    """
    Spatial Modulation Adapter (SMA):
    Calculates pixel-wise attention masks to focus on semantic regions and suppress background noise.
    """
    def __init__(self, channels):
        super().__init__()
        # Use depthwise-separable convolution to minimize parameter overhead
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        mask = self.conv(x)
        return x * mask

class FMA(nn.Module):
    """
    Frequency Modulation Adapter (FMA):
    Performs spatial 2D-FFT per channel, modulates frequency components,
    and performs inverse FFT to reconstruct spatial features.
    """
    def __init__(self, channels):
        super().__init__()
        # 1x1 convolution acts as a lightweight frequency-component gating network
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 1. Forward Spatial 2D-FFT
        x_freq = torch.fft.rfft2(x, norm="ortho")
        
        # 2. Extract frequency magnitude
        mag = torch.abs(x_freq)
        
        # 3. Predict frequency modulation weights dynamically
        mod = self.conv(mag)
        
        # 4. Apply modulation
        x_freq_modulated = x_freq * mod
        
        # 5. Inverse Spatial 2D-FFT
        x_recon = torch.fft.irfft2(x_freq_modulated, s=x.shape[-2:], norm="ortho")
        return x_recon

class SSFMA(nn.Module):
    """
    Spectral-Spatial-Frequency Modulation Adapter (SSFMA):
    Sequentially links SpMA, SMA, and FMA to progressively refine latent representations.
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.spma = SpMA(channels, reduction=reduction)
        self.sma = SMA(channels)
        self.fma = FMA(channels)

    def forward(self, x):
        x = self.spma(x)
        x = self.sma(x)
        x = self.fma(x)
        return x
