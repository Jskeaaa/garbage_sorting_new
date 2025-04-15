import os
import argparse
import torch
from tqdm import tqdm

from config import SEED, MODEL_SAVE_DIR
from utils import set_seed, create_dir_if_not_exists
from data_preparation import get_data_loaders
from train import main as train_main
from evaluate import main as evaluate_main
from predict import main as predict_main


def main():
    """主函数，处理命令行参数并执行相应的功能"""
    parser = argparse.ArgumentParser(description='垃圾分类系统')
    
    # 创建子命令解析器
    subparsers = parser.add_subparsers(dest='command', help='要执行的命令')
    
    # 训练命令
    train_parser = subparsers.add_parser('train', help='训练模型')
    
    # 评估命令
    eval_parser = subparsers.add_parser('evaluate', help='评估模型')
    eval_parser.add_argument('--model_path', type=str, default=None, 
                            help='模型路径，默认使用saved_models/model_best.pth')
    
    # 预测命令
    predict_parser = subparsers.add_parser('predict', help='预测单张图像')
    predict_parser.add_argument('--image_path', type=str, required=True, 
                              help='待预测图像的路径')
    predict_parser.add_argument('--model_path', type=str, default=None, 
                              help='模型路径，默认使用saved_models/model_best.pth')
    predict_parser.add_argument('--visualize', action='store_true', 
                              help='是否可视化结果')
    predict_parser.add_argument('--heatmap', action='store_true', 
                              help='是否生成热图')
    
    # 批量预测命令
    batch_predict_parser = subparsers.add_parser('batch_predict', help='批量预测多张图像')
    batch_predict_parser.add_argument('--image_dir', type=str, required=True, 
                                    help='包含待预测图像的目录')
    batch_predict_parser.add_argument('--model_path', type=str, default=None, 
                                    help='模型路径，默认使用saved_models/model_best.pth')
    batch_predict_parser.add_argument('--output_file', type=str, default='predictions.csv', 
                                    help='预测结果输出文件')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(SEED)
    
    # 确保保存目录存在
    create_dir_if_not_exists(MODEL_SAVE_DIR)
    
    # 根据命令执行相应的功能
    if args.command == 'train':
        print("开始训练模型...")
        train_main()
    
    elif args.command == 'evaluate':
        print("开始评估模型...")
        model_path = args.model_path
        if model_path is None:
            model_path = os.path.join(MODEL_SAVE_DIR, 'model_best.pth')
            if not os.path.exists(model_path):
                print(f"找不到默认模型文件: {model_path}")
                print("请先训练模型或指定正确的模型路径")
                return
        
        evaluate_main(model_path)
    
    elif args.command == 'predict':
        print("开始预测图像...")
        image_path = args.image_path
        model_path = args.model_path
        if model_path is None:
            model_path = os.path.join(MODEL_SAVE_DIR, 'model_best.pth')
            if not os.path.exists(model_path):
                print(f"找不到默认模型文件: {model_path}")
                print("请先训练模型或指定正确的模型路径")
                return
        
        predict_main(
            image_path=image_path,
            model_path=model_path,
            visualize=args.visualize,
            generate_heatmap=args.heatmap
        )
    
    elif args.command == 'batch_predict':
        print("开始批量预测图像...")
        image_dir = args.image_dir
        model_path = args.model_path
        output_file = args.output_file
        
        if model_path is None:
            model_path = os.path.join(MODEL_SAVE_DIR, 'model_best.pth')
            if not os.path.exists(model_path):
                print(f"找不到默认模型文件: {model_path}")
                print("请先训练模型或指定正确的模型路径")
                return
        
        batch_predict(
            image_dir=image_dir,
            model_path=model_path,
            output_file=output_file
        )
    
    else:
        parser.print_help()


def batch_predict(image_dir, model_path, output_file):
    """批量预测目录中的所有图像"""
    import csv
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载模型
    from model import get_model
    model = get_model(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    # 获取图像文件列表
    image_files = []
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(root, file))
    
    print(f"找到 {len(image_files)} 张图像")
    
    # 批量预测
    results = []
    for image_file in tqdm(image_files, desc="预测中"):
        try:
            # 加载并预处理图像
            from predict import load_image, predict_image
            _, img_tensor = load_image(image_file)
            
            # 预测
            result = predict_image(model, img_tensor, device)
            
            # 记录结果
            results.append({
                'image_path': image_file,
                'predicted_class': result['class_name'],
                'probability': result['probability']
            })
        except Exception as e:
            print(f"处理图像 {image_file} 时出错: {str(e)}")
            results.append({
                'image_path': image_file,
                'predicted_class': 'ERROR',
                'probability': 0.0
            })
    
    # 保存结果到CSV文件
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['image_path', 'predicted_class', 'probability'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"预测结果已保存到: {output_file}")


if __name__ == "__main__":
    main() 