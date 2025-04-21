import torch
import torch.nn as nn
import torchvision.models as models
from config import NUM_CLASSES, PRETRAINED

class GarbageClassifier(nn.Module):
    """基于ResNet-50的垃圾分类模型"""
    
    def __init__(self, num_classes=NUM_CLASSES, pretrained=PRETRAINED):
        super(GarbageClassifier, self).__init__()
        
        # 加载预训练的ResNet-50模型
        self.model = models.resnet50(pretrained=pretrained)
        
        # 获取特征维度
        in_features = self.model.fc.in_features
        
        # 替换最后的全连接层
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),  # 添加Dropout以减少过拟合
            nn.Linear(in_features, num_classes)
        )
        
        # 初始化新添加的层
        nn.init.xavier_uniform_(self.model.fc[1].weight)
        nn.init.zeros_(self.model.fc[1].bias)
    
    def forward(self, x):
        """前向传播"""
        return self.model(x)
    
    def freeze_backbone(self):
        """冻结骨干网络参数，只训练分类头"""
        for param in self.model.parameters():
            param.requires_grad = False
        
        # 解冻最后一个块和全连接层
        for param in self.model.layer4.parameters():
            param.requires_grad = True
        
        for param in self.model.fc.parameters():
            param.requires_grad = True
    
    def unfreeze_backbone(self):
        """解冻所有层，进行微调"""
        for param in self.model.parameters():
            param.requires_grad = True
    
    def get_trainable_params(self):
        """返回需要训练的参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_params(self):
        """返回模型总参数数量"""
        return sum(p.numel() for p in self.parameters())


def get_model(device):
    """获取模型实例并移动到指定设备"""
    model = GarbageClassifier(num_classes=NUM_CLASSES, pretrained=PRETRAINED)
    model = model.to(device)
    return model


if __name__ == "__main__":
    # 创建模型实例
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(device)
    
    # 打印模型信息
    print(f"模型总参数数量: {model.get_total_params():,}")
    print(f"可训练参数数量: {model.get_trainable_params():,}")
    
    # 测试冻结和解冻功能
    model.freeze_backbone()
    print(f"冻结后可训练参数数量: {model.get_trainable_params():,}")
    
    model.unfreeze_backbone()
    print(f"解冻后可训练参数数量: {model.get_trainable_params():,}")
    
    # 测试前向传播
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    output = model(dummy_input)
    print(f"输出形状: {output.shape}") 