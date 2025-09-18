# bee_gui.py

# 设置 Matplotlib 支持中文
import matplotlib
matplotlib.use('Qt5Agg')
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'FangSong', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号

import sys
import os
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QListWidget, QSplitter,
    QTextEdit, QGroupBox, QScrollArea, QGridLayout, QMessageBox
)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt
from ultralytics import YOLO
import numpy as np
import pandas as pd

# =================== 配置 ===================
MODEL_PATH = r"D:\PythonProject\runs\detect\bee_detection3\weights\best.pt"  # 修改为您的 best.pt 路径
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

# 加载模型
model = YOLO(MODEL_PATH)


class BeeDetectionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🐝 蜜蜂智能检测系统")
        self.resize(1200, 800)

        self.image_list = []
        self.current_image = None
        self.result_image = None
        self.detection_results = []  # 存储所有检测结果（用于导出）

        self.init_ui()

    def init_ui(self):
        # 主布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget)

        # 左侧面板：图像列表 + 统计
        left_panel = QSplitter(Qt.Vertical)
        layout.addWidget(left_panel, 1)

        # 图像列表
        self.image_list_widget = QListWidget()
        self.image_list_widget.itemClicked.connect(self.on_image_click)
        list_group = QGroupBox("图像列表")
        list_layout = QVBoxLayout()
        list_layout.addWidget(self.image_list_widget)
        load_btn = QPushButton("📁 加载图像目录")
        load_btn.clicked.connect(self.load_images)
        list_layout.addWidget(load_btn)
        list_group.setLayout(list_layout)
        left_panel.addWidget(list_group)

        # 统计信息
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setFont(QFont("Consolas", 10))
        stats_group = QGroupBox("📊 当前目录统计")
        stats_layout = QVBoxLayout()
        stats_layout.addWidget(self.stats_text)
        stats_group.setLayout(stats_layout)
        left_panel.addWidget(stats_group)

        # 右侧面板：图像显示 + 控制
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # 图像显示区域（可滚动）
        scroll_area = QScrollArea()
        self.image_label = QLabel("请加载图像")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(400, 300)
        scroll_area.setWidget(self.image_label)
        scroll_area.setWidgetResizable(True)
        right_layout.addWidget(scroll_area)

        # 控制按钮
        btn_layout = QGridLayout()
        self.detect_btn = QPushButton("🔍 检测当前图像")
        self.detect_btn.clicked.connect(self.detect_current)
        self.detect_btn.setEnabled(False)

        self.batch_btn = QPushButton("📁 批量检测")
        self.batch_btn.clicked.connect(self.batch_detect)

        self.save_btn = QPushButton("💾 保存结果")
        self.save_btn.clicked.connect(self.save_result)
        self.save_btn.setEnabled(False)

        btn_layout.addWidget(self.detect_btn, 0, 0)
        btn_layout.addWidget(self.batch_btn, 0, 1)
        btn_layout.addWidget(self.save_btn, 1, 0, 1, 2)

        right_layout.addLayout(btn_layout)

        # 检测结果信息
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setFont(QFont("Consolas", 10))
        result_group = QGroupBox("📈 检测结果")
        result_layout = QVBoxLayout()
        result_layout.addWidget(self.result_text)
        result_group.setLayout(result_layout)
        right_layout.addWidget(result_group)

        layout.addWidget(right_panel, 2)

    def load_images(self):
        """加载指定目录下的图像文件到列表，并更新统计"""
        dir_path = QFileDialog.getExistingDirectory(self, "选择图像目录")
        if not dir_path:
            return

        self.image_list = []
        self.image_list_widget.clear()

        p = Path(dir_path)
        for ext in IMAGE_EXTENSIONS:
            self.image_list.extend(p.rglob(f"*{ext}"))

        for img_path in self.image_list:
            self.image_list_widget.addItem(str(img_path.relative_to(p)))

        # 更新当前目录的统计信息
        self.update_current_dir_stats(dir_path)

    def update_current_dir_stats(self, dir_path):
        """更新当前加载目录的统计信息"""
        dir_path = Path(dir_path)
        if not dir_path.exists():
            self.stats_text.setPlainText("❌ 目录不存在")
            return

        # 统计各格式图像数量
        image_count = {}
        total_count = 0
        for ext in IMAGE_EXTENSIONS:
            files = list(dir_path.rglob(f"*{ext}"))
            if files:
                image_count[ext.upper()] = len(files)
                total_count += len(files)

        # 构建统计文本
        stats_text = f"📁 当前目录: {dir_path}\n"
        stats_text += f"========================\n"
        if total_count == 0:
            stats_text += "⚠️  该目录中未找到支持的图像文件"
        else:
            for ext, count in image_count.items():
                stats_text += f"{ext} 图像: {count}\n"
            stats_text += f"------------------------\n"
            stats_text += f"✅ 总图像数: {total_count}\n"

        # 添加模型信息
        stats_text += f"\n🧠 模型: {Path(MODEL_PATH).name}"

        self.stats_text.setPlainText(stats_text)

    def on_image_click(self, item):
        """点击图像列表项时加载图像"""
        try:
            relative_path = item.text()
            base_dir = self.image_list[0].parent if self.image_list else Path(".")
            full_path = base_dir / relative_path
            self.current_image = str(full_path)
            self.load_and_show_image(full_path)
            self.detect_btn.setEnabled(True)
            self.result_text.clear()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载图像失败: {e}")

    def load_and_show_image(self, image_path):
        """加载并显示图像"""
        image = cv2.imread(str(image_path))
        if image is None:
            self.image_label.setText("❌ 图像加载失败")
            return

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = image.shape
        bytes_per_line = ch * w
        qt_image = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image).scaled(800, 600, Qt.KeepAspectRatio)
        self.image_label.setPixmap(pixmap)
        self.image_label.setText("")
        self.current_image = str(image_path)

    def detect_current(self):
        """检测当前选中的单张图像"""
        if not self.current_image:
            return

        try:
            results = model(self.current_image, conf=0.25)
            result = results[0]

            # 在图像上绘制结果
            annotated_frame = result.plot()
            self.result_image = annotated_frame

            # 转换为 QPixmap 显示
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = annotated_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(annotated_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image).scaled(800, 600, Qt.KeepAspectRatio)
            self.image_label.setPixmap(pixmap)

            # 输出检测信息
            boxes = result.boxes
            num_detections = len(boxes)
            avg_conf = boxes.conf.mean().item() if len(boxes) > 0 else 0

            self.result_text.clear()
            self.result_text.append(f"🎯 检测结果:\n")
            self.result_text.append(f"  检测到蜜蜂数量: {num_detections}")
            self.result_text.append(f"  平均置信度: {avg_conf:.3f}")
            self.result_text.append(f"  模型: {Path(MODEL_PATH).name}")

            # 保存本次检测结果
            self.detection_results.append({
                'image': self.current_image,
                'detections': num_detections,
                'confidence': avg_conf
            })

            self.save_btn.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, "检测错误", f"检测失败: {e}")

    def batch_detect(self):
        """用户选择多张图像进行批量检测"""
        options = QFileDialog.Options()
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "选择要检测的图像",
            "",  # 初始目录（可设为上次路径）
            "Image Files (*.jpg *.jpeg *.png *.bmp *.webp);;All Files (*)",
            options=options
        )

        if not file_paths:
            return  # 用户取消选择

        total_detections = 0
        total_conf = 0.0
        count = 0  # 有检测结果的图像数量
        batch_results = []

        # 处理每张选中的图像
        for img_path in file_paths:
            try:
                results = model(img_path, conf=0.25)
                result = results[0]
                boxes = result.boxes
                num_detections = len(boxes)
                avg_conf = boxes.conf.mean().item() if num_detections > 0 else 0.0

                batch_results.append({
                    'image': img_path,
                    'detections': num_detections,
                    'confidence': avg_conf
                })

                total_detections += num_detections
                if num_detections > 0:
                    total_conf += avg_conf
                    count += 1

            except Exception as e:
                print(f"处理图像失败: {img_path}, 错误: {e}")
                continue

        # 将本次批量检测结果加入全局列表
        self.detection_results.extend(batch_results)

        # 计算统计信息
        avg_conf = total_conf / count if count > 0 else 0
        avg_per_image = total_detections / len(file_paths)

        # 弹窗提示完成
        QMessageBox.information(
            self, "批量检测完成",
            f"📊 共检测 {len(file_paths)} 张图像\n"
            f"  总检测数量: {total_detections}\n"
            f"  平均每张图像: {avg_per_image:.2f} 只蜜蜂\n"
            f"  平均置信度: {avg_conf:.3f}"
        )

        # 显示分布图
        self.show_detection_distribution()

    def show_detection_distribution(self):
        """显示检测数量分布图"""
        if not self.detection_results:
            return

        detections = [r['detections'] for r in self.detection_results]
        plt.figure(figsize=(8, 5))
        plt.hist(detections, bins=range(max(detections) + 2), edgecolor='black', alpha=0.7)
        plt.xlabel('检测到的蜜蜂数量')
        plt.ylabel('图像数量')
        plt.title('检测数量分布图')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()

    def save_result(self):
        """将所有检测结果保存为 CSV 文件"""
        if not self.detection_results:
            QMessageBox.warning(self, "提示", "暂无检测结果可保存")
            return

        df = pd.DataFrame(self.detection_results)
        save_path, _ = QFileDialog.getSaveFileName(
            self, "保存检测结果", "", "CSV Files (*.csv)"
        )
        if save_path:
            if not save_path.endswith('.csv'):
                save_path += '.csv'
            df.to_csv(save_path, index=False)
            QMessageBox.information(self, "保存成功", f"结果已保存至:\n{save_path}")


# ============ 启动应用 ============
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BeeDetectionGUI()
    window.show()
    sys.exit(app.exec_())