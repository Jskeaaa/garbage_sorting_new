import os

# 数据集路径
DATA_DIR = os.path.join(os.getcwd(), 'Garbage classification')

# 类别信息
CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
NUM_CLASSES = len(CLASSES)

# 数据集分割比例
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# 图像预处理参数
IMG_SIZE = 224  # ResNet-50要求的输入大小
MEAN = [0.485, 0.456, 0.406]  # ImageNet预训练模型的归一化参数
STD = [0.229, 0.224, 0.225]

# 训练参数
BATCH_SIZE = 32
NUM_EPOCHS = 30
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
MOMENTUM = 0.9
EARLY_STOPPING_PATIENCE = 5
SCHEDULER_PATIENCE = 2
SCHEDULER_FACTOR = 0.1
NUM_WORKERS = 0  # 数据加载器的工作进程数，0表示只使用主进程，可避免序列化问题

# 模型参数
FREEZE_BACKBONE = True  # 是否冻结backbone
FREEZE_EPOCHS = 5  # 冻结backbone的epoch数量
PRETRAINED = True  # 是否使用预训练模型

# 保存路径
MODEL_SAVE_DIR = 'saved_models'
LOG_DIR = 'runs'

# 随机种子
SEED = 42

# 数据增强参数
AUG_SCALE = (0.8, 1.0)
AUG_RATIO = (0.75, 1.33)
AUG_BRIGHTNESS = 0.2
AUG_CONTRAST = 0.2
AUG_SATURATION = 0.2
AUG_HUE = 0.1
AUG_PROB = 0.5  # 应用数据增强的概率 