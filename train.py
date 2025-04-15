import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import (
    NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, MOMENTUM, EARLY_STOPPING_PATIENCE,
    SCHEDULER_PATIENCE, SCHEDULER_FACTOR, FREEZE_BACKBONE, FREEZE_EPOCHS,
    MODEL_SAVE_DIR, LOG_DIR, SEED
)
from utils import (
    set_seed, save_checkpoint, create_dir_if_not_exists, get_lr,
    plot_loss_acc_curves
)
from data_preparation import get_data_loaders
from model import get_model


def train_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    epoch_loss = 0
    epoch_corrects = 0
    total_samples = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        # 统计
        epoch_loss += loss.item() * inputs.size(0)
        epoch_corrects += torch.sum(preds == labels.data)
        total_samples += inputs.size(0)
        
        # 更新进度条
        pbar.set_postfix(loss=loss.item(), acc=torch.sum(preds == labels.data).item()/inputs.size(0))
    
    # 计算平均损失和准确率
    epoch_loss = epoch_loss / total_samples
    epoch_acc = epoch_corrects.double() / total_samples
    
    return epoch_loss, epoch_acc.item()


def validate_epoch(model, val_loader, criterion, device):
    """验证一个epoch"""
    model.eval()
    epoch_loss = 0
    epoch_corrects = 0
    total_samples = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            # 统计
            epoch_loss += loss.item() * inputs.size(0)
            epoch_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)
            
            # 更新进度条
            pbar.set_postfix(loss=loss.item(), acc=torch.sum(preds == labels.data).item()/inputs.size(0))
    
    # 计算平均损失和准确率
    epoch_loss = epoch_loss / total_samples
    epoch_acc = epoch_corrects.double() / total_samples
    
    return epoch_loss, epoch_acc.item()


def train_model(model, dataloaders, criterion, optimizer, scheduler, device, class_weights=None, num_epochs=NUM_EPOCHS):
    """训练模型的主函数"""
    # 创建保存模型的目录
    create_dir_if_not_exists(MODEL_SAVE_DIR)
    create_dir_if_not_exists(LOG_DIR)
    
    # 获取数据加载器
    train_loader, val_loader = dataloaders
    
    # 设置TensorBoard日志
    log_dir = os.path.join(LOG_DIR, time.strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(log_dir=log_dir)
    
    # 保存最佳模型
    best_model_params = None
    best_acc = 0.0
    best_epoch = 0
    
    # 用于早停的计数器
    early_stopping_counter = 0
    
    # 记录训练历史
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    # 设置冻结/解冻策略
    if FREEZE_BACKBONE:
        print("冻结骨干网络...")
        model.freeze_backbone()
    
    # 打印可训练参数数量
    print(f"可训练参数数量: {model.get_trainable_params():,}")
    
    # 训练循环
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # 是否解冻模型
        if FREEZE_BACKBONE and epoch == FREEZE_EPOCHS:
            print("解冻骨干网络进行微调...")
            model.unfreeze_backbone()
            print(f"可训练参数数量: {model.get_trainable_params():,}")
        
        # 训练阶段
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}")
        
        # 验证阶段
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        # 记录历史
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # 记录到TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('Learning_rate', get_lr(optimizer), epoch)
        
        # 更新学习率
        if scheduler:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            best_model_params = model.state_dict().copy()
            
            # 保存检查点
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }, True, os.path.join(MODEL_SAVE_DIR, 'checkpoint.pth'),
            os.path.join(MODEL_SAVE_DIR, 'model_best.pth'))
            
            # 重置早停计数器
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            print(f"EarlyStopping counter: {early_stopping_counter} out of {EARLY_STOPPING_PATIENCE}")
            
            if early_stopping_counter >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered after epoch {epoch+1}")
                break
    
    # 训练结束，加载最佳模型
    print(f"\n训练完成！最佳验证准确率: {best_acc:.4f} 在第 {best_epoch+1} 轮")
    model.load_state_dict(best_model_params)
    
    # 绘制损失和准确率曲线
    plot_loss_acc_curves(train_losses, val_losses, train_accs, val_accs, 
                         save_path=os.path.join(MODEL_SAVE_DIR, 'training_curves.png'))
    
    # 关闭TensorBoard写入器
    writer.close()
    
    return model


def main():
    """主函数"""
    # 设置随机种子，确保结果可复现
    set_seed(SEED)
    
    # 确定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载数据
    train_loader, val_loader, test_loader, class_weights = get_data_loaders()
    print(f"训练集样本数: {len(train_loader.dataset)}")
    print(f"验证集样本数: {len(val_loader.dataset)}")
    print(f"测试集样本数: {len(test_loader.dataset)}")
    print(f"类别权重: {class_weights}")
    
    # 创建模型
    model = get_model(device)
    print(f"模型总参数数量: {model.get_total_params():,}")
    
    # 定义损失函数（使用类别权重处理不平衡问题）
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # 定义优化器
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, 
                           momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    
    # 定义学习率调度器（在验证损失不下降时减小学习率）
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=SCHEDULER_FACTOR,
                                  patience=SCHEDULER_PATIENCE, verbose=True)
    
    # 训练模型
    trained_model = train_model(
        model=model,
        dataloaders=(train_loader, val_loader),
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        class_weights=class_weights,
        num_epochs=NUM_EPOCHS
    )
    
    return trained_model, test_loader, device


if __name__ == "__main__":
    main() 