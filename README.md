# 垃圾分类系统

基于ResNet-50的垃圾分类系统，可以将垃圾图像分类为6个类别：纸板(cardboard)、玻璃(glass)、金属(metal)、纸张(paper)、塑料(plastic)和其他垃圾(trash)。

## 数据集

数据集位于`Garbage classification`文件夹中，包含6个类别的图像：
- cardboard（纸板）：403张图像
- glass（玻璃）：501张图像
- metal（金属）：410张图像
- paper（纸张）：594张图像
- plastic（塑料）：482张图像
- trash（其他垃圾）：137张图像

## 项目特点

- 基于预训练的ResNet-50模型进行迁移学习
- 采用分层训练策略，先冻结骨干网络，再微调
- 使用数据增强提高模型泛化能力
- 处理数据集不平衡问题，提高小类别的识别准确率
- 提供可视化工具，包括混淆矩阵、训练曲线和类激活映射(CAM)

## 环境要求

本项目基于Python 3.6+和PyTorch 1.8+实现，主要依赖包包括：

```
torch>=1.8.0
torchvision>=0.9.0
numpy>=1.19.5
matplotlib>=3.3.4
scikit-learn>=0.24.1
tqdm>=4.60.0
Pillow>=8.2.0
pandas>=1.2.4
tensorboard>=2.5.0
```

## 安装

1. 克隆项目到本地
2. 安装依赖包：

```bash
pip install -r requirements.txt
```

## 使用方法

### 训练模型

```bash
python main.py train
```

训练参数可在`config.py`文件中修改。

### 评估模型

```bash
python main.py evaluate [--model_path MODEL_PATH]
```

默认使用`saved_models/model_best.pth`模型文件。

### 预测单张图像

```bash
python main.py predict --image_path IMAGE_PATH [--model_path MODEL_PATH] [--visualize] [--heatmap]
```

参数说明：
- `--image_path`：待预测图像的路径（必需）
- `--model_path`：模型路径（可选，默认使用`saved_models/model_best.pth`）
- `--visualize`：是否可视化结果（可选）
- `--heatmap`：是否生成热图（可选）

### 批量预测多张图像

```bash
python main.py batch_predict --image_dir IMAGE_DIR [--model_path MODEL_PATH] [--output_file OUTPUT_FILE]
```

参数说明：
- `--image_dir`：包含待预测图像的目录（必需）
- `--model_path`：模型路径（可选，默认使用`saved_models/model_best.pth`）
- `--output_file`：预测结果输出文件（可选，默认为`predictions.csv`）

## 项目结构

```
garbage_sorting/
├── Garbage classification/  # 数据集
├── config.py                # 配置文件
├── data_preparation.py      # 数据预处理和加载
├── evaluate.py              # 模型评估
├── main.py                  # 主入口点
├── model.py                 # 模型定义
├── predict.py               # 单张图像预测
├── README.md                # 项目说明
├── requirements.txt         # 依赖包列表
├── train.py                 # 模型训练
└── utils.py                 # 工具函数
```

## 模型性能

在测试集上的预期性能：
- 准确率：> 90%
- 每个类别的准确率：
  - cardboard: > 92%
  - glass: > 91%
  - metal: > 90%
  - paper: > 95%
  - plastic: > 85%
  - trash: > 80%

## 示例输出

![预测结果示例](saved_models/prediction_example.png)

## 模型权重

训练好的模型权重将保存在`saved_models`目录中。

## 可视化

训练过程可视化：
```bash
tensorboard --logdir=runs
```

## 注意事项

- 初次运行时会下载预训练的ResNet-50模型，确保网络连接正常
- 如果GPU内存不足，可在`config.py`中调小批次大小（BATCH_SIZE）
- 对于小型数据集，建议增大数据增强参数或使用更小的学习率 