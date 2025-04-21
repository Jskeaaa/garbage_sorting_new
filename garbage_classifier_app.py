

import os
import sys
import torch
import numpy as np
from PIL import Image, ImageQt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                            QStatusBar, QAction, QMenu, QMessageBox, QProgressBar,
                            QSplitter, QFrame, QGridLayout, QGroupBox)
from PyQt5.QtGui import QPixmap, QImage, QIcon, QPalette, QColor
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize, QTimer

# 导入现有项目中的模块
from config import CLASSES, IMG_SIZE, MEAN, STD, MODEL_SAVE_DIR
from model import get_model
from utils import set_seed, generate_activation_map
from predict import load_image, predict_image


class PredictionWorker(QThread):
    """后台线程用于执行模型预测，避免阻塞UI"""
    
    # 定义信号
    prediction_complete = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)
    
    def __init__(self, model, image_path, device, show_heatmap=False):
        super().__init__()
        self.model = model
        self.image_path = image_path
        self.device = device
        self.show_heatmap = show_heatmap
    
    def run(self):
        try:
            # 加载并预处理图像
            self.progress.emit(20)
            img, img_tensor = load_image(self.image_path)
            
            # 预测
            self.progress.emit(50)
            result = predict_image(self.model, img_tensor, self.device)
            
            # 生成热图（如果需要）
            self.progress.emit(70)
            if self.show_heatmap:
                target_layer = self.model.model.layer4[-1]
                heatmap = generate_activation_map(
                    self.model, img_tensor, target_layer, result['class_idx']
                )
                
                # 调整热图尺寸以匹配原图
                heatmap = Image.fromarray(np.uint8(heatmap * 255))
                heatmap = heatmap.resize((img.size[0], img.size[1]), Image.BICUBIC)
                heatmap = np.array(heatmap) / 255.0
                
                # 添加热图到结果
                result['heatmap'] = heatmap
            
            # 添加原始图像到结果
            result['image'] = img
            
            self.progress.emit(100)
            self.prediction_complete.emit(result)
            
        except Exception as e:
            self.error.emit(str(e))


class GarbageClassifierApp(QMainWindow):
    """垃圾分类应用程序主窗口"""
    
    def __init__(self):
        super().__init__()
        
        # 设置窗口属性
        self.setWindowTitle("垃圾分类系统")
        self.setMinimumSize(1000, 700)
        
        # 初始化变量
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.current_image_path = None
        self.show_heatmap = False
        
        # 初始化UI
        self.init_ui()
        
        # 加载模型
        self.load_model()
    
    def init_ui(self):
        """初始化UI组件"""
        # 中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout(central_widget)
        
        # 创建分割器
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        
        # 左侧图像显示区域
        self.image_frame = QGroupBox("图像预览")
        image_layout = QVBoxLayout()
        self.image_frame.setLayout(image_layout)
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(400, 400)
        self.image_label.setFrameShape(QFrame.Box)
        self.image_label.setStyleSheet("border: 1px solid #cccccc; background-color: #f2f2f2;")
        self.image_label.setText("请选择图像")
        
        image_layout.addWidget(self.image_label)
        
        # 按钮区域
        button_layout = QHBoxLayout()
        self.open_button = QPushButton("打开图像")
        self.open_button.clicked.connect(self.open_image)
        
        self.predict_button = QPushButton("开始分类")
        self.predict_button.clicked.connect(self.predict_image)
        self.predict_button.setEnabled(False)
        
        self.heatmap_button = QPushButton("显示热图")
        self.heatmap_button.setCheckable(True)
        self.heatmap_button.toggled.connect(self.toggle_heatmap)
        self.heatmap_button.setEnabled(False)
        
        button_layout.addWidget(self.open_button)
        button_layout.addWidget(self.predict_button)
        button_layout.addWidget(self.heatmap_button)
        
        image_layout.addLayout(button_layout)
        
        # 右侧结果显示区域
        self.result_frame = QGroupBox("分类结果")
        result_layout = QVBoxLayout()
        self.result_frame.setLayout(result_layout)
        
        self.result_label = QLabel("请先选择图像并进行分类")
        self.result_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.result_label.setWordWrap(True)
        self.result_label.setStyleSheet("font-size: 14px;")
        
        # 添加各类别概率显示区域
        self.probabilities_layout = QGridLayout()
        result_layout.addWidget(self.result_label)
        result_layout.addSpacing(20)
        result_layout.addLayout(self.probabilities_layout)
        
        # 创建概率标签
        self.probability_labels = {}
        self.probability_bars = {}
        
        for i, class_name in enumerate(CLASSES):
            # 类别标签
            class_label = QLabel(class_name)
            class_label.setStyleSheet("font-weight: bold;")
            
            # 概率值标签
            prob_label = QLabel("0.00")
            
            # 概率条
            prob_bar = QProgressBar()
            prob_bar.setRange(0, 100)
            prob_bar.setValue(0)
            
            # 添加到布局
            self.probabilities_layout.addWidget(class_label, i, 0)
            self.probabilities_layout.addWidget(prob_bar, i, 1)
            self.probabilities_layout.addWidget(prob_label, i, 2)
            
            # 保存引用
            self.probability_labels[class_name] = prob_label
            self.probability_bars[class_name] = prob_bar
        
        result_layout.addStretch()
        
        # 添加到分割器
        splitter.addWidget(self.image_frame)
        splitter.addWidget(self.result_frame)
        
        # 设置初始分割比例
        splitter.setSizes([500, 500])
        
        # 添加到主布局
        main_layout.addWidget(splitter)
        
        # 状态栏
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("就绪")
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setVisible(False)
        self.statusBar.addPermanentWidget(self.progress_bar)
        
        # 菜单栏
        menubar = self.menuBar()
        
        # 文件菜单
        file_menu = menubar.addMenu("文件")
        
        open_action = QAction("打开图像", self)
        open_action.triggered.connect(self.open_image)
        open_action.setShortcut("Ctrl+O")
        file_menu.addAction(open_action)
        
        exit_action = QAction("退出", self)
        exit_action.triggered.connect(self.close)
        exit_action.setShortcut("Ctrl+Q")
        file_menu.addAction(exit_action)
        
        # 帮助菜单
        help_menu = menubar.addMenu("帮助")
        
        about_action = QAction("关于", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def load_model(self):
        """加载预训练的模型"""
        try:
            self.statusBar.showMessage("正在加载模型...")
            
            model_path = os.path.join(MODEL_SAVE_DIR, 'model_best.pth')
            
            if not os.path.exists(model_path):
                QMessageBox.critical(self, "错误", "模型文件不存在，请先训练模型或指定正确的模型路径")
                self.statusBar.showMessage("模型加载失败")
                return
            
            self.model = get_model(self.device)
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.model.eval()
            
            self.statusBar.showMessage(f"模型加载成功，使用设备: {self.device}")
        
        except Exception as e:
            QMessageBox.critical(self, "错误", f"模型加载失败: {str(e)}")
            self.statusBar.showMessage("模型加载失败")
    
    def open_image(self):
        """打开图像文件"""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图像", "", "图像文件 (*.jpg *.jpeg *.png);;所有文件 (*)",
            options=options
        )
        
        if file_path:
            try:
                # 加载图像
                self.current_image_path = file_path
                
                # 显示图像
                pixmap = QPixmap(file_path)
                pixmap = pixmap.scaled(self.image_label.width(), self.image_label.height(), 
                                      Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.image_label.setPixmap(pixmap)
                
                # 更新状态
                self.predict_button.setEnabled(True)
                self.statusBar.showMessage(f"已加载图像: {os.path.basename(file_path)}")
                
                # 清除旧的结果
                self.result_label.setText("请点击'开始分类'按钮进行预测")
                
                # 重置概率条
                for class_name in CLASSES:
                    self.probability_labels[class_name].setText("0.00")
                    self.probability_bars[class_name].setValue(0)
                
                self.heatmap_button.setEnabled(False)
                self.heatmap_button.setChecked(False)
                
            except Exception as e:
                QMessageBox.critical(self, "错误", f"无法打开图像: {str(e)}")
                self.statusBar.showMessage("图像加载失败")
    
    def predict_image(self):
        """预测图像类别"""
        if not self.current_image_path or not self.model:
            return
        
        # 禁用按钮，防止重复点击
        self.predict_button.setEnabled(False)
        self.open_button.setEnabled(False)
        
        # 显示进度条
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.statusBar.showMessage("正在进行预测...")
        
        # 创建并启动工作线程
        self.worker = PredictionWorker(
            self.model, self.current_image_path, self.device, self.show_heatmap
        )
        self.worker.prediction_complete.connect(self.show_prediction_result)
        self.worker.error.connect(self.show_prediction_error)
        self.worker.progress.connect(self.update_progress)
        self.worker.start()
    
    def show_prediction_result(self, result):
        """显示预测结果"""
        # 更新结果文本
        result_text = f"预测类别: <b>{result['class_name']}</b><br>"
        result_text += f"置信度: <b>{result['probability']:.2f}</b><br><br>"
        
        self.result_label.setText(result_text)
        
        # 更新概率条
        all_classes = [cls for cls, _ in result['all_probs']]
        all_probs = [prob for _, prob in result['all_probs']]
        
        # 创建字典映射类别到概率
        probs_dict = dict(zip(all_classes, all_probs))
        
        # 更新所有类别的概率
        for class_name in CLASSES:
            prob = probs_dict.get(class_name, 0.0)
            self.probability_labels[class_name].setText(f"{prob:.2f}")
            self.probability_bars[class_name].setValue(int(prob * 100))
        
        # 如果有热图且显示热图开关打开，则显示热图
        if 'heatmap' in result and self.show_heatmap:
            self.show_heatmap_on_image(result['image'], result['heatmap'])
            self.heatmap_button.setEnabled(True)
        elif 'image' in result:
            # 显示原始图像
            pil_image = result['image']
            q_image = self.pil_to_qimage(pil_image)
            pixmap = QPixmap.fromImage(q_image)
            pixmap = pixmap.scaled(self.image_label.width(), self.image_label.height(), 
                                  Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(pixmap)
            
            # 如果已计算了热图，启用热图按钮
            if 'heatmap' in result:
                # 保存热图和图像到实例变量，而不是尝试访问self.result
                self.heatmap_button.setEnabled(True)
        
        # 恢复按钮状态
        self.predict_button.setEnabled(True)
        self.open_button.setEnabled(True)
        
        # 隐藏进度条并更新状态
        self.progress_bar.setVisible(False)
        self.statusBar.showMessage("预测完成")
        
        # 保存结果以便切换热图显示
        self.result = result
    
    def show_prediction_error(self, error_message):
        """显示预测错误"""
        QMessageBox.critical(self, "预测错误", error_message)
        
        # 恢复按钮状态
        self.predict_button.setEnabled(True)
        self.open_button.setEnabled(True)
        
        # 隐藏进度条并更新状态
        self.progress_bar.setVisible(False)
        self.statusBar.showMessage("预测失败")
    
    def update_progress(self, value):
        """更新进度条"""
        self.progress_bar.setValue(value)
    
    def toggle_heatmap(self, checked):
        """切换热图显示"""
        self.show_heatmap = checked
        
        if hasattr(self, 'result'):
            if checked and 'heatmap' in self.result:
                self.show_heatmap_on_image(self.result['image'], self.result['heatmap'])
            elif 'image' in self.result:
                # 显示原始图像
                pil_image = self.result['image']
                q_image = self.pil_to_qimage(pil_image)
                pixmap = QPixmap.fromImage(q_image)
                pixmap = pixmap.scaled(self.image_label.width(), self.image_label.height(), 
                                      Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.image_label.setPixmap(pixmap)
    
    def pil_to_qimage(self, pil_image):
        """将PIL图像转换为QImage"""
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        
        # 转换为QImage
        data = pil_image.tobytes("raw", "RGB")
        q_image = QImage(data, pil_image.size[0], pil_image.size[1], pil_image.size[0] * 3, QImage.Format_RGB888)
        return q_image
    
    def show_heatmap_on_image(self, original_image, heatmap):
        """在原始图像上叠加热图显示"""
        # 将热图转换为彩色图像
        heatmap_rgb = np.zeros((heatmap.shape[0], heatmap.shape[1], 3), dtype=np.uint8)
        for i in range(heatmap.shape[0]):
            for j in range(heatmap.shape[1]):
                # 使用红色表示热区
                heatmap_rgb[i, j, 0] = int(255 * heatmap[i, j])  # 红色通道
        
        # 转换为PIL图像
        heatmap_image = Image.fromarray(heatmap_rgb)
        
        # 叠加热图和原图
        blended = Image.blend(original_image.convert("RGB"), heatmap_image, 0.5)
        
        # 显示图像
        q_image = self.pil_to_qimage(blended)
        pixmap = QPixmap.fromImage(q_image)
        pixmap = pixmap.scaled(self.image_label.width(), self.image_label.height(), 
                              Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(pixmap)
    
    def show_about(self):
        """显示关于对话框"""
        QMessageBox.about(self, "关于", 
                          "垃圾分类系统\n\n"
                          "基于ResNet-50的垃圾分类系统，可以将垃圾图像分类为6个类别：\n"
                          "纸板(cardboard)、玻璃(glass)、金属(metal)、纸张(paper)、塑料(plastic)和其他垃圾(trash)。\n\n"
                          "开发者: Pmocoding")
    
    def resizeEvent(self, event):
        """窗口大小改变事件"""
        # 如果有图像，重新缩放图像
        if self.image_label.pixmap() and not self.image_label.pixmap().isNull():
            pixmap = self.image_label.pixmap().scaled(
                self.image_label.width(), self.image_label.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.image_label.setPixmap(pixmap)
        
        super().resizeEvent(event)


if __name__ == "__main__":
    # 设置随机种子
    set_seed(42)
    
    # 创建应用
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # 使用Fusion风格，在所有平台上看起来都不错
    
    # 创建主窗口
    window = GarbageClassifierApp()
    window.show()
    
    # 运行应用
    sys.exit(app.exec_()) 