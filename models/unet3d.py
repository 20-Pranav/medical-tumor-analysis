import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool3d(2)

    def forward(self, x):
        f = self.conv(x)
        p = self.pool(f)
        return f, p

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        # Calculate padding to match skip connection size
        diff_z = skip.size(2) - x.size(2)
        diff_y = skip.size(3) - x.size(3)
        diff_x = skip.size(4) - x.size(4)
        x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2,
                      diff_y // 2, diff_y - diff_y // 2,
                      diff_z // 2, diff_z - diff_z // 2])
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x

class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=32):
        super().__init__()
        self.enc1 = Encoder(in_channels, init_features)
        self.enc2 = Encoder(init_features, init_features*2)
        self.enc3 = Encoder(init_features*2, init_features*4)
        self.enc4 = Encoder(init_features*4, init_features*8)
        self.bottleneck = ConvBlock(init_features*8, init_features*16)
        self.dec4 = Decoder(init_features*16, init_features*8)
        self.dec3 = Decoder(init_features*8, init_features*4)
        self.dec2 = Decoder(init_features*4, init_features*2)
        self.dec1 = Decoder(init_features*2, init_features)
        self.out_conv = nn.Conv3d(init_features, out_channels, kernel_size=1)

    def forward(self, x):
        # x shape: (B, C, D, H, W)
        f1, p1 = self.enc1(x)
        f2, p2 = self.enc2(p1)
        f3, p3 = self.enc3(p2)
        f4, p4 = self.enc4(p3)
        b = self.bottleneck(p4)
        d4 = self.dec4(b, f4)
        d3 = self.dec3(d4, f3)
        d2 = self.dec2(d3, f2)
        d1 = self.dec1(d2, f1)
        out = torch.sigmoid(self.out_conv(d1))
        return out
