import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import numpy as np
from config import (
    DATA_DIR, CLASSES, IMG_SIZE, MEAN, STD, BATCH_SIZE,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO, AUG_SCALE, AUG_RATIO,
    AUG_BRIGHTNESS, AUG_CONTRAST, AUG_SATURATION, AUG_HUE, AUG_PROB,
    NUM_WORKERS
)
from utils import set_seed

# 删除本地的NUM_WORKERS定义，使用config.py中的定义


class GarbageDataset(Dataset):
    """垃圾分类数据集"""
    
    def __init__(self, data_dir, transform=None):
        """
        初始化垃圾分类数据集
        
        Args:
            data_dir (str): 数据集路径
            transform (callable, optional): 应用于图像的转换
        """
        self.data_dir = data_dir
        self.transform = transform
        self.classes = CLASSES
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        self.samples = []
        self._load_samples()
    
    def _load_samples(self):
        """加载数据集中的所有样本"""
        for class_name in self.classes:
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, self.class_to_idx[class_name]))
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """获取指定索引的样本"""
        img_path, label = self.samples[idx]
        
        # 打开图像
        try:
            img = Image.open(img_path).convert('RGB')
        except:
            # 如果图像打开失败，返回一个默认图像和标签
            img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), color='black')
            print(f"Warning: Could not open image {img_path}")
        
        # 应用变换
        if self.transform:
            img = self.transform(img)
        
        return img, label


# 将TransformedSubset类移到模块级别
class TransformedSubset(Dataset):
    """转换后的数据子集，用于对数据子集应用不同的变换"""
    
    def __init__(self, subset, transform, full_dataset):
        """
        初始化转换后的数据子集
        
        Args:
            subset: 数据子集
            transform: 应用的变换
            full_dataset: 完整数据集，用于获取图像路径和标签
        """
        self.subset = subset
        self.transform = transform
        self.full_dataset = full_dataset
    
    def __len__(self):
        """返回数据子集大小"""
        return len(self.subset)
    
    def __getitem__(self, idx):
        """获取指定索引的样本并应用变换"""
        img_path, label = self.full_dataset.samples[self.subset.indices[idx]]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


def get_data_loaders():
    """获取训练集、验证集和测试集的数据加载器"""
    # 数据增强-训练集
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=AUG_SCALE, ratio=AUG_RATIO),
        transforms.RandomHorizontalFlip(p=AUG_PROB),
        transforms.RandomVerticalFlip(p=AUG_PROB),
        transforms.RandomRotation(15),
        transforms.ColorJitter(
            brightness=AUG_BRIGHTNESS,
            contrast=AUG_CONTRAST,
            saturation=AUG_SATURATION,
            hue=AUG_HUE
        ),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])
    
    # 标准变换-验证集和测试集
    val_test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])
    
    # 创建完整数据集
    full_dataset = GarbageDataset(DATA_DIR, transform=None)
    
    # 获取数据集大小
    total_size = len(full_dataset)
    
    # 计算各子集大小
    train_size = int(TRAIN_RATIO * total_size)
    val_size = int(VAL_RATIO * total_size)
    test_size = total_size - train_size - val_size
    
    # 分割数据集
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # 应用不同的变换
    train_dataset = TransformedSubset(train_dataset, train_transform, full_dataset)
    val_dataset = TransformedSubset(val_dataset, val_test_transform, full_dataset)
    test_dataset = TransformedSubset(test_dataset, val_test_transform, full_dataset)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
    )
    
    # 返回数据加载器和类别权重（类别不平衡处理）
    class_counts = [0] * len(CLASSES)
    for _, label in full_dataset.samples:
        class_counts[label] += 1
    
    # 计算类别权重
    total_samples = sum(class_counts)
    class_weights = [total_samples / (len(CLASSES) * count) for count in class_counts]
    class_weights = torch.FloatTensor(class_weights)
    
    return train_loader, val_loader, test_loader, class_weights


if __name__ == "__main__":
    # 设置随机种子
    set_seed(42)
    
    # 获取数据加载器
    train_loader, val_loader, test_loader, class_weights = get_data_loaders()
    
    # 打印数据集信息
    print(f"训练集样本数: {len(train_loader.dataset)}")
    print(f"验证集样本数: {len(val_loader.dataset)}")
    print(f"测试集样本数: {len(test_loader.dataset)}")
    print(f"类别权重: {class_weights}")
    
    # 查看一个批次的数据形状
    for images, labels in train_loader:
        print(f"批次图像形状: {images.shape}")
        print(f"批次标签形状: {labels.shape}")
        break 