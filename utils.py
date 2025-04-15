import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import itertools
import torch.nn.functional as F
from config import CLASSES

def set_seed(seed):
    """设置随机种子以确保结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def compute_class_weights(dataset, num_classes):
    """计算类别权重，用于处理不平衡数据集"""
    # 统计每个类别的样本数量
    class_counts = [0] * num_classes
    for _, target in dataset:
        class_counts[target] += 1
    
    # 计算权重：样本数量的倒数，归一化
    weights = [1.0 / count for count in class_counts]
    total = sum(weights)
    weights = [weight / total * num_classes for weight in weights]
    
    return torch.FloatTensor(weights)

def plot_loss_acc_curves(train_losses, val_losses, train_accs, val_accs, save_path=None):
    """绘制训练和验证的损失和准确率曲线"""
    plt.figure(figsize=(12, 5))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curves')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(len(CLASSES))
    plt.xticks(tick_marks, CLASSES, rotation=45)
    plt.yticks(tick_marks, CLASSES)
    
    # 在矩阵中显示数字
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def print_classification_report(y_true, y_pred):
    """打印分类报告"""
    report = classification_report(y_true, y_pred, target_names=CLASSES)
    print(report)

def generate_activation_map(model, img_tensor, target_layer, class_idx):
    """生成类激活映射 (CAM) 可视化"""
    # 注意：这只是一个简化的CAM实现，适用于ResNet这类使用Global Average Pooling的网络
    
    # 获取模型特定层的输出
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    
    # 注册钩子以获取目标层的输出
    handle = target_layer.register_forward_hook(get_activation('target_layer'))
    
    # 通过模型传递图像，获取输出
    model.eval()
    output = model(img_tensor.unsqueeze(0))
    
    # 获取分类器权重
    weights = model.fc.weight.data[class_idx].cpu().numpy()
    
    # 获取特征图
    feature_maps = activation['target_layer'].squeeze(0).cpu().numpy()
    
    # 计算CAM
    cam = np.zeros(feature_maps.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * feature_maps[i, :, :]
    
    # 应用ReLU，只保留正值
    cam = np.maximum(cam, 0)
    
    # 归一化
    if cam.max() != 0:
        cam = cam / cam.max()
    
    # 移除钩子
    handle.remove()
    
    return cam

def get_lr(optimizer):
    """获取优化器的当前学习率"""
    for param_group in optimizer.param_groups:
        return param_group['lr']

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', best_filename='model_best.pth.tar'):
    """保存模型检查点"""
    torch.save(state, filename)
    if is_best:
        torch.save(state, best_filename)

def create_dir_if_not_exists(dir_path):
    """如果目录不存在，则创建目录"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path) 