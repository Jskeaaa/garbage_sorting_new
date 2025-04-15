import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

from config import CLASSES, IMG_SIZE, MEAN, STD, MODEL_SAVE_DIR
from model import get_model
from utils import set_seed, generate_activation_map


def load_image(image_path):
    """加载并预处理图像"""
    # 图像变换
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])
    
    # 加载图像
    try:
        img = Image.open(image_path).convert('RGB')
    except:
        raise ValueError(f"无法打开图像: {image_path}")
    
    # 应用变换
    img_tensor = transform(img)
    
    return img, img_tensor


def predict_image(model, image_tensor, device, top_k=3):
    """预测图像类别"""
    # 设置模型为评估模式
    model.eval()
    
    # 将图像移动到设备
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    # 预测
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
    
    # 获取top-k类别和概率
    top_probs, top_indices = torch.topk(probabilities, top_k)
    
    # 转换为numpy
    top_probs = top_probs.cpu().numpy()
    top_indices = top_indices.cpu().numpy()
    
    # 获取类别名称
    top_classes = [CLASSES[idx] for idx in top_indices]
    
    # 返回结果
    result = {
        'class_idx': top_indices[0],  # 最高概率的类别索引
        'class_name': top_classes[0],  # 最高概率的类别名称
        'probability': top_probs[0],   # 最高概率值
        'all_probs': [(cls, prob) for cls, prob in zip(top_classes, top_probs)]  # 所有top-k结果
    }
    
    return result


def visualize_prediction(img, result, heatmap=None):
    """可视化预测结果"""
    plt.figure(figsize=(12, 6))
    
    if heatmap is not None:
        # 绘制原始图像
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title(f"预测: {result['class_name']} ({result['probability']:.2f})")
        plt.axis('off')
        
        # 绘制热图
        plt.subplot(1, 2, 2)
        plt.imshow(img)
        plt.imshow(heatmap, cmap='jet', alpha=0.5)
        plt.title("类激活热图")
        plt.axis('off')
    else:
        # 绘制原始图像
        plt.imshow(img)
        plt.title(f"预测: {result['class_name']} ({result['probability']:.2f})")
        plt.axis('off')
    
    # 绘制概率条形图
    plt.figure(figsize=(10, 6))
    classes = [cls for cls, _ in result['all_probs']]
    probs = [prob for _, prob in result['all_probs']]
    
    plt.barh(classes, probs)
    plt.xlabel('概率')
    plt.ylabel('类别')
    plt.title('预测概率分布')
    plt.xlim(0, 1)
    plt.tight_layout()
    
    plt.show()


def main(image_path, model_path=None, visualize=True, generate_heatmap=False):
    """主函数"""
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载模型
    if model_path is None:
        model_path = os.path.join(MODEL_SAVE_DIR, 'model_best.pth')
    
    if not os.path.exists(model_path):
        raise ValueError(f"模型文件不存在: {model_path}")
    
    print(f"加载模型从: {model_path}")
    model = get_model(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    
    # 加载并预处理图像
    img, img_tensor = load_image(image_path)
    
    # 预测
    result = predict_image(model, img_tensor, device)
    
    # 打印结果
    print(f"\n预测结果:")
    print(f"类别: {result['class_name']}")
    print(f"概率: {result['probability']:.4f}")
    print("\n其他可能的类别:")
    for cls, prob in result['all_probs'][1:]:
        print(f"{cls}: {prob:.4f}")
    
    # 生成热图
    heatmap = None
    if generate_heatmap:
        # 获取最后一层卷积层来生成类激活映射
        target_layer = model.model.layer4[-1]
        heatmap = generate_activation_map(
            model, img_tensor, target_layer, result['class_idx']
        )
        
        # 调整热图尺寸以匹配原图
        heatmap = Image.fromarray(np.uint8(heatmap * 255))
        heatmap = heatmap.resize((img.size[0], img.size[1]), Image.BICUBIC)
        heatmap = np.array(heatmap) / 255.0
    
    # 可视化
    if visualize:
        visualize_prediction(img, result, heatmap)
    
    return result


if __name__ == "__main__":
    import argparse
    
    # 命令行参数
    parser = argparse.ArgumentParser(description='垃圾分类预测')
    parser.add_argument('--image_path', type=str, required=True, help='待预测图像的路径')
    parser.add_argument('--model_path', type=str, default=None, help='模型路径')
    parser.add_argument('--visualize', action='store_true', help='是否可视化结果')
    parser.add_argument('--heatmap', action='store_true', help='是否生成热图')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(42)
    
    # 执行预测
    main(args.image_path, args.model_path, args.visualize, args.heatmap) 