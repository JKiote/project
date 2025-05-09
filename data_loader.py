import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, num_samples=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = []
        
        # 提前过滤损坏文件
        for filename in os.listdir(root_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                filepath = os.path.join(root_dir, filename)
                try:
                    with Image.open(filepath) as img:
                        img.verify()  # 验证文件完整性
                    self.image_files.append(filepath)
                except Exception as e:
                    print(f"Skipped corrupted file: {filepath} ({e})")
        
        if num_samples:
            self.image_files = self.image_files[:num_samples]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

def load_data(data_path, batch_size=32, transform=None, num_samples=None):
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    dataset = CustomImageDataset(data_path, transform, num_samples)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)  # 增加num_workers