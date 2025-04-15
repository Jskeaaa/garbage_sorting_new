import os
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from config import CLASSES, MODEL_SAVE_DIR
from utils import set_seed, plot_confusion_matrix, print_classification_report, create_dir_if_not_exists
from model import get_model
from train import main as train_main


def evaluate_model(model, test_loader, device):
    """评估模型在测试集上的性能"""
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            # 收集预测结果和真实标签
            all_predictions.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
    
    # 计算评估指标
    accuracy = accuracy_score(all_targets, all_predictions)
    print(f"\n测试集准确率: {accuracy:.4f}")
    
    # 打印分类报告
    print("\n分类报告:")
    print_classification_report(all_targets, all_predictions)
    
    # 计算并可视化混淆矩阵
    print("\n混淆矩阵:")
    plot_confusion_matrix(
        all_targets, all_predictions,
        save_path=os.path.join(MODEL_SAVE_DIR, 'confusion_matrix.png')
    )
    
    # 按类别统计准确率
    class_correct = [0] * len(CLASSES)
    class_total = [0] * len(CLASSES)
    
    for target, pred in zip(all_targets, all_predictions):
        class_correct[target] += (target == pred)
        class_total[target] += 1
    
    # 打印每个类别的准确率
    print("\n按类别的准确率:")
    for i in range(len(CLASSES)):
        if class_total[i] > 0:
            print(f"{CLASSES[i]}: {class_correct[i]/class_total[i]:.4f} ({class_correct[i]}/{class_total[i]})")
    
    return accuracy, all_targets, all_predictions


def analyze_errors(model, test_loader, device, num_samples=5):
    """分析错误分类的样本"""
    model.eval()
    error_samples = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            # 找出错误分类的样本
            error_indices = (preds != labels).nonzero(as_tuple=True)[0]
            for idx in error_indices:
                # 收集错误样本的信息
                error_samples.append({
                    'image': inputs[idx].cpu(),
                    'true_label': labels[idx].item(),
                    'pred_label': preds[idx].item(),
                    'confidence': torch.softmax(outputs[idx], dim=0)[preds[idx]].item()
                })
                
                # 如果收集了足够多的样本，就返回
                if len(error_samples) >= num_samples:
                    return error_samples
    
    return error_samples


def load_model_for_evaluation(model_path, device):
    """加载模型用于评估"""
    # 创建一个新的模型实例
    model = get_model(device)
    
    # 加载模型参数
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    
    return model


def main(model_path=None):
    """主函数"""
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 训练模型或加载已有模型
    if model_path is None:
        print("没有指定模型路径，开始训练模型...")
        model, test_loader, device = train_main()
    else:
        # 从train.py导入函数获取测试数据加载器
        from train import main as train_main
        from data_preparation import get_data_loaders
        
        # 获取数据加载器
        train_loader, val_loader, test_loader, _ = get_data_loaders()
        
        # 加载模型
        print(f"加载模型从路径: {model_path}")
        model = load_model_for_evaluation(model_path, device)
    
    # 评估模型
    print("开始评估模型...")
    accuracy, targets, predictions = evaluate_model(model, test_loader, device)
    
    # 分析错误
    print("\n分析错误样本...")
    error_samples = analyze_errors(model, test_loader, device)
    print(f"找到 {len(error_samples)} 个错误分类的样本")
    
    # 打印一些错误样本的信息
    for i, sample in enumerate(error_samples):
        print(f"错误样本 {i+1}:")
        print(f"  真实标签: {CLASSES[sample['true_label']]}")
        print(f"  预测标签: {CLASSES[sample['pred_label']]}")
        print(f"  置信度: {sample['confidence']:.4f}")
    
    return accuracy


if __name__ == "__main__":
    # 确保结果目录存在
    create_dir_if_not_exists(MODEL_SAVE_DIR)
    
    # 设置随机种子
    set_seed(42)
    
    # 指定模型路径或设为None以训练新模型
    model_path = os.path.join(MODEL_SAVE_DIR, 'model_best.pth')
    if not os.path.exists(model_path):
        model_path = None
    
    # 评估模型
    main(model_path) 