import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

# --------------------- 自注意力模块 ---------------------
class SelfAttention(nn.Module):
    # 保持原有实现不变
    def __init__(self, in_channels):
        super().__init__()
        self.query = spectral_norm(nn.Conv2d(in_channels, in_channels//8, 1))
        self.key = spectral_norm(nn.Conv2d(in_channels, in_channels//8, 1))
        self.value = spectral_norm(nn.Conv2d(in_channels, in_channels, 1))
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # 保持原有实现不变
        batch_size, C, H, W = x.size()
        query = self.query(x).view(batch_size, -1, H*W).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, H*W)
        energy = torch.bmm(query, key)
        attention = torch.softmax(energy, dim=-1)
        value = self.value(x).view(batch_size, -1, H*W)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, H, W)
        return self.gamma * out + x

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            # 初始输入：100 -> 512x4x4
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            # 阶段1：4x4 -> 8x8
            nn.Conv2d(512, 512*4, 3, padding=1),
            nn.PixelShuffle(2),  # 512*4 -> 512
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            ResidualBlock(512),
            
            # 阶段2：8x8 -> 16x16
            nn.Conv2d(512, 256*4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            SelfAttention(256),
            
            # 阶段3：16x16 -> 32x32
            nn.Conv2d(256, 128*4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            # 阶段4：32x32 -> 64x64
            nn.Conv2d(128, 64*4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # 新增阶段5：64x64 -> 128x128
            nn.Conv2d(64, 32*4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(32),  # 新增
            nn.LeakyReLU(0.2),   # 新增
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

class ResidualBlock(nn.Module):
    """残差块定义"""
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels)
        )
    
    def forward(self, x):
        return x + self.block(x)

# --------------------- 判别器（保持原结构） ---------------------
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.global_branch = nn.Sequential(
            spectral_norm(nn.Conv2d(3, 32, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(32, 128, 4, 2, 1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            SelfAttention(128),
            spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)),  # 新增层以处理更大尺寸
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )
        self.local_branch = nn.Sequential(
            spectral_norm(nn.Conv2d(3, 64, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)),  # 新增层以处理更大尺寸
            nn.LeakyReLU(0.2)
        )
        self.final = nn.Sequential(
            spectral_norm(nn.Conv2d(512, 512, 4, 2, 1)),  # 调整输入通道数
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(512, 1, 4, 1, 0))
        )

    def forward(self, x):
        global_feat = self.global_branch(x)
        local_feat = self.local_branch(x[:, :, 32:96, 32:96]) 
        local_feat = nn.functional.interpolate(local_feat, size=global_feat.shape[2:], mode='bilinear')
        combined = torch.cat([global_feat, local_feat], dim=1)
        return self.final(combined).view(-1)