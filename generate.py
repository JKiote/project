import torch
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from models import Generator

# 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载生成器模型
z_dim = 100
generator = Generator().to(device)
generator.load_state_dict(torch.load('generator.pth'))
generator.eval()

# 生成一些动漫头像进行可视化
num_samples = 16
noise = torch.randn(num_samples, z_dim, 1, 1).to(device)
with torch.no_grad():
    generated_images = generator(noise).cpu()

# 反归一化处理
generated_images = torch.clamp((generated_images + 1) / 2.0, 0.0, 1.0)

# 可视化生成的图像
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
axes = axes.flatten()
for i in range(num_samples):
    img = np.transpose(generated_images[i], (1, 2, 0))
    axes[i].imshow(img)
    axes[i].axis('off')
plt.show()