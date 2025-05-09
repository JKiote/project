import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from data_loader import load_data
from models import Generator, Discriminator
from torch.autograd import grad
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from torch_fidelity import calculate_metrics

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 加载数据
    data_path = 'F:/teset/data/raw'
    num_samples =40000
    dataloader = load_data(data_path, transform=transform, num_samples=num_samples)

    # 初始化模型
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    # 优化器
    lr = 0.0001
    beta1, beta2 = 0.5, 0.999
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))

    # 训练参数
    num_epochs = 50
    z_dim = 100
    num_critic = 5
    lambda_gp = 10

    # 学习率调度器
    scheduler_G = lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=num_epochs)
    scheduler_D = lr_scheduler.CosineAnnealingLR(optimizer_D, T_max=num_epochs)

    # 损失记录
    d_losses, g_losses = [], []

    # 创建目录
    os.makedirs('models', exist_ok=True)
    os.makedirs('generated_final', exist_ok=True)  # 最终生成图像目录
    os.makedirs('real_images', exist_ok=True)
    os.makedirs('preview_images', exist_ok=True)  # 预览图像目录

    # 保存真实图像（仅运行一次）
    real_images_path = 'real_images'
    if len(os.listdir(real_images_path)) == 0:
        for i, real_images in enumerate(dataloader):
            if i * dataloader.batch_size >= 1000:
                break
            for j in range(real_images.size(0)):
                img = (real_images[j].permute(1, 2, 0).cpu().numpy() + 1) / 2
                plt.imsave(f'{real_images_path}/real_{i*dataloader.batch_size+j}.png', (img*255).astype('uint8'))

    # 训练循环
    for epoch in range(num_epochs):
        d_epoch_loss = 0
        g_epoch_loss = 0
        loop = tqdm(enumerate(dataloader), total=len(dataloader), leave=False)
        
        for i, (real_images) in loop:
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            # 训练判别器
            for _ in range(num_critic):
                optimizer_D.zero_grad()
                real_output = discriminator(real_images).view(-1)
                
                noise = torch.randn(batch_size, z_dim, 1, 1).to(device)
                fake_images = generator(noise).detach()
                fake_output = discriminator(fake_images).view(-1)
                
                # 梯度惩罚
                alpha = torch.rand(batch_size, 1, 1, 1).to(device)
                interpolated = alpha * real_images + (1 - alpha) * fake_images
                interpolated.requires_grad = True
                interpolated_output = discriminator(interpolated).view(-1)
                
                gradients = grad(
                    outputs=interpolated_output, inputs=interpolated,
                    grad_outputs=torch.ones_like(interpolated_output),
                    create_graph=True, retain_graph=True
                )[0]
                gradient_penalty = ((gradients.norm(2, dim=1) - 1)**2).mean() * lambda_gp
                
                d_loss = -(torch.mean(real_output) - torch.mean(fake_output)) + gradient_penalty
                d_loss.backward()
                optimizer_D.step()
                d_epoch_loss += d_loss.item()

            # 训练生成器
            optimizer_G.zero_grad()
            noise = torch.randn(batch_size, z_dim, 1, 1).to(device)
            fake_output = discriminator(generator(noise)).view(-1)
            g_loss = -torch.mean(fake_output)
            g_loss.backward()
            optimizer_G.step()
            g_epoch_loss += g_loss.item()

            loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")

        # 记录损失
        d_losses.append(d_epoch_loss / len(dataloader))
        g_losses.append(g_epoch_loss / len(dataloader))
        print(f'Epoch [{epoch+1}/{num_epochs}] D_loss: {d_losses[-1]:.4f} G_loss: {g_losses[-1]:.4f}')

        # 更新学习率
        scheduler_G.step()
        scheduler_D.step()

        # 每隔5个epoch生成50张图片预览
        if (epoch + 1) % 5 == 0:
            preview_noise = torch.randn(50, z_dim, 1, 1).to(device)
            with torch.no_grad():
                preview_images = generator(preview_noise).cpu()
                preview_dir = f'preview_images/epoch_{epoch+1}'
                os.makedirs(preview_dir, exist_ok=True)
                for i in range(50):
                    img = (preview_images[i].permute(1, 2, 0).numpy() * 127.5 + 127.5).astype('uint8')
                    plt.imsave(f'{preview_dir}/preview_{i}.png', img)

    # ==== 训练结束后生成图像并计算FID ====
    print("\n训练完成，开始生成最终图像并计算FID...")
    fixed_noise = torch.randn(1000, z_dim, 1, 1).to(device)
    with torch.no_grad():
        fake_images = generator(fixed_noise).cpu()
        fake_dir = 'generated_final/final_1000'
        os.makedirs(fake_dir, exist_ok=True)
        for i in range(1000):
            img = (fake_images[i].permute(1, 2, 0).numpy() * 127.5 + 127.5).astype('uint8')
            plt.imsave(f'{fake_dir}/fake_{i}.png', img)
        
        # 计算FID
        metrics = calculate_metrics(
            input1=real_images_path,
            input2=fake_dir,
            fid=True,
            cuda=device.type == 'cuda'
        )
        final_fid = metrics['frechet_inception_distance']
        print(f"\n最终FID值: {final_fid:.4f}")

    # 保存模型
    torch.save({
        'generator': generator.state_dict(),
        'discriminator': discriminator.state_dict(),
        'final_fid': final_fid,
        'd_losses': d_losses,
        'g_losses': g_losses
    }, 'models/final_model.pth')

    # 绘制损失曲线
    plt.plot(d_losses, label='D Loss')
    plt.plot(g_losses, label='G Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_curve.png')
    plt.show()

if __name__ == '__main__':
    main()