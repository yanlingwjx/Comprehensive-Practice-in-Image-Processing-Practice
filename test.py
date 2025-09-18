import os
import yaml
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple


class BeeDetector:
    """
    蜜蜂检测器类，用于训练和推理YOLOv8模型检测野外图片中的蜜蜂
    """

    def __init__(self, data_path: str, model_config: str = "yolov8.yaml",
                 model_name: str = "yolov8n.pt"):
        """
        初始化蜜蜂检测器

        Args:
            data_path: 数据集路径，包含 images/ 和 labels/ 目录
            model_config: 模型配置文件路径
            model_name: 预训练模型名称
        """
        self.data_path = Path(data_path)
        self.model_config = model_config
        self.model_name = model_name
        self.model = None
        self.data_yaml = None

        # 检查数据集结构
        self._check_dataset_structure()

    def _check_dataset_structure(self):
        """✅ MODIFIED: 检查数据集结构是否符合要求"""
        # ✅ MODIFIED: 顶层目录是 images/ 和 labels/
        if not (self.data_path / 'images').exists():
            raise FileNotFoundError(f"数据集目录缺少 'images' 目录")

        if not (self.data_path / 'labels').exists():
            raise FileNotFoundError(f"数据集目录缺少 'labels' 目录")

        # ✅ MODIFIED: 检查 images/train, images/val, images/test
        splits = ['train', 'val', 'test']  # 注意：YOLO 使用 'val'，不是 'valid'
        for split in splits:
            images_path = self.data_path / 'images' / split
            labels_path = self.data_path / 'labels' / split
            if not images_path.exists():
                raise FileNotFoundError(f"缺少图片目录: {images_path}")
            if not labels_path.exists():
                raise FileNotFoundError(f"缺少标签目录: {labels_path}")

    def create_data_yaml(self) -> str:
        """
        ✅ MODIFIED: 创建YOLO格式的数据配置文件

        Returns:
            data_yaml文件路径
        """
        data_yaml = {
            'train': str(self.data_path / 'images' / 'train'),  # ✅ MODIFIED: 指向 images/train
            'val': str(self.data_path / 'images' / 'val'),      # ✅ MODIFIED: 指向 images/val
            'test': str(self.data_path / 'images' / 'test'),    # ✅ MODIFIED: 指向 images/test
            'nc': 1,  # 类别数量，蜜蜂为1类
            'names': ['bee']  # 类别名称
        }

        yaml_path = self.data_path / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True)

        self.data_yaml = str(yaml_path)
        print(f"数据配置文件已创建: {self.data_yaml}")
        return self.data_yaml

    def load_model(self, pretrained: bool = True):
        """
        加载YOLOv8模型

        Args:
            pretrained: 是否使用预训练权重
        """
        if pretrained:
            self.model = YOLO(self.model_name)
            print(f"已加载预训练模型: {self.model_name}")
        else:
            self.model = YOLO(self.model_config)
            print(f"已加载自定义模型配置: {self.model_config}")

    def train(self, epochs: int = 100, imgsz: int = 640, batch: int = 16,
              save_dir: str = "runs/train"):
        """
        训练模型

        Args:
            epochs: 训练轮数
            imgsz: 图像尺寸
            batch: 批次大小
            save_dir: 保存目录
        """
        if not self.model:
            self.load_model()

        if not self.data_yaml:
            self.create_data_yaml()

        print("开始训练模型...")
        results = self.model.train(
            data=self.data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            save_dir=save_dir,
            name='bee_detection',
            patience=20,  # 早停耐心值
            optimizer='AdamW',  # 优化器
            lr0=0.001,  # 初始学习率
            lrf=0.01,  # 最终学习率
            momentum=0.937,  # SGD动量/Adam beta1
            weight_decay=0.0005,  # 权重衰减
            warmup_epochs=3.0,  # 热身轮数
            warmup_momentum=0.8,  # 热身动量
            warmup_bias_lr=0.1,  # 热身偏置学习率
            box=7.5,  # 损失函数中边界框损失的权重
            cls=0.5,  # 损失函数中类别损失的权重
            dfl=1.5,  # 损失函数中dfl损失的权重
            close_mosaic=10,  # 关闭mosaic增强的轮数
        )

        print("训练完成!")
        return results

    def validate(self):
        """
        验证模型性能
        """
        if not self.model:
            # 加载最佳模型
            model_path = "runs/train/bee_detection/weights/best.pt"
            if os.path.exists(model_path):
                self.model = YOLO(model_path)
            else:
                raise FileNotFoundError("未找到训练好的模型权重文件")

        print("开始验证模型...")
        results = self.model.val()
        print(f"验证完成: mAP50-95 = {results.box.map50:.4f}")
        return results

    def predict(self, source: str, conf_threshold: float = 0.5) -> List[Dict]:
        """
        对单张图片或图片目录进行预测

        Args:
            source: 图片路径或目录路径
            conf_threshold: 置信度阈值

        Returns:
            包含检测结果的列表
        """
        if not self.model:
            # 尝试加载最佳模型
            model_path = "runs/train/bee_detection/weights/best.pt"
            if os.path.exists(model_path):
                self.model = YOLO(model_path)
            else:
                raise FileNotFoundError("未找到训练好的模型权重文件")

        # 进行预测
        results = self.model(source, conf=conf_threshold, imgsz=640)

        detections = []
        for result in results:
            image_info = {
                'path': result.path,
                'height': result.orig_shape[0],
                'width': result.orig_shape[1],
                'boxes': []
            }

            # 提取边界框信息
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf.cpu().numpy()[0]
                cls = int(box.cls.cpu().numpy()[0])

                image_info['boxes'].append({
                    'class': 'bee',
                    'confidence': float(conf),
                    'bbox': [float(x1), float(y1), float(x2), float(y2)]
                })

            detections.append(image_info)

        return detections

    def visualize_detections(self, detections: List[Dict], output_dir: str = "output"):
        """
        可视化检测结果

        Args:
            detections: 检测结果列表
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)

        for detection in detections:
            # 读取原始图像
            image = cv2.imread(detection['path'])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 绘制边界框和标签
            for box_info in detection['boxes']:
                bbox = box_info['bbox']
                conf = box_info['confidence']
                x1, y1, x2, y2 = map(int, bbox)

                # 绘制边界框
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # 添加标签
                label = f"bee: {conf:.2f}"
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)

            # 显示蜜蜂数量
            bee_count = len(detection['boxes'])
            cv2.putText(image, f"Bees: {bee_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # 保存结果图像
            filename = Path(detection['path']).stem + '_detected.jpg'
            output_path = os.path.join(output_dir, filename)
            plt.figure(figsize=(12, 8))
            plt.imshow(image)
            plt.axis('off')
            plt.title(f"Detected Bees: {bee_count}")
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"检测结果已保存: {output_path}")

    def evaluate_test_set(self, conf_threshold: float = 0.5):
        """
        评估测试集性能

        Args:
            conf_threshold: 置信度阈值
        """
        test_images_path = self.data_path / 'images' / 'test'  # ✅ MODIFIED: 指向 images/test

        print(f"开始评估测试集: {test_images_path}")
        detections = self.predict(str(test_images_path), conf_threshold)

        # 统计每张图片的蜜蜂数量
        results_summary = []
        total_bees = 0

        for detection in detections:
            bee_count = len(detection['boxes'])
            results_summary.append({
                'image': Path(detection['path']).name,
                'bee_count': bee_count,
                'boxes': detection['boxes']
            })
            total_bees += bee_count

        # 打印统计信息
        print(f"测试集评估完成!")
        print(f"共处理 {len(detections)} 张图片")
        print(f"共检测到 {total_bees} 只蜜蜂")
        print(f"平均每张图片检测到 {total_bees / len(detections):.2f} 只蜜蜂")

        # 保存结果到文件
        results_file = "test_results_summary.txt"
        with open(results_file, 'w') as f:
            f.write(f"测试集评估结果\n")
            f.write(f"时间: {os.popen('date').read().strip()}\n")  # Windows/Linux 兼容性可能有问题
            f.write(f"置信度阈值: {conf_threshold}\n")
            f.write(f"共处理 {len(detections)} 张图片\n")
            f.write(f"共检测到 {total_bees} 只蜜蜂\n")
            f.write(f"平均每张图片检测到 {total_bees / len(detections):.2f} 只蜜蜂\n")
            f.write("\n详细结果:\n")
            for result in results_summary:
                f.write(f"{result['image']}: {result['bee_count']} bees\n")

        print(f"评估结果已保存到: {results_file}")

        return results_summary


def main():
    """
    主函数：演示如何使用蜜蜂检测器
    """
    # ✅ MODIFIED: 设置数据集路径（请根据实际情况修改）
    # 示例结构：
    # D:\PythonProject\yolo_dataset\
    # ├── images\
    # │   ├── train
    # │   ├── val
    # │   └── test
    # ├── labels\
    # │   ├── train
    # │   ├── val
    # │   └── test
    # └── data.yaml
    data_path = r"D:\PythonProject\yolo_dataset"  # 修改为您的实际路径

    # 创建蜜蜂检测器实例
    bee_detector = BeeDetector(data_path)

    # 步骤1: 创建数据配置文件
    # ✅ 注意：如果已有 data.yaml，可以注释掉此行
    # bee_detector.create_data_yaml()

    # 步骤2: 加载模型
    bee_detector.load_model(pretrained=True)

    # 步骤3: 训练模型
    print("是否开始训练？(y/n): ")
    if input().lower() == 'y':
        bee_detector.train(epochs=30, batch=16)

        # 步骤4: 验证模型
        bee_detector.validate()

    # 步骤5: 评估测试集
    print("是否评估测试集？(y/n): ")
    if input().lower() == 'y':
        results = bee_detector.evaluate_test_set(conf_threshold=0.5)

    # 步骤6: 对单张图片进行预测
    print("是否对示例图片进行预测？(y/n): ")
    if input().lower() == 'y':
        # 使用测试集中的第一张图片作为示例
        test_images_path = bee_detector.data_path / 'images' / 'test'
        sample_images = list(test_images_path.glob('*.jpg'))
        if sample_images:
            sample_image = sample_images[0]
            detections = bee_detector.predict(str(sample_image), conf_threshold=0.5)

            # 显示检测结果
            for detection in detections:
                bee_count = len(detection['boxes'])
                print(f"图片: {detection['path']}")
                print(f"检测到 {bee_count} 只蜜蜂:")
                for i, box in enumerate(detection['boxes']):
                    print(f"  蜜蜂 {i + 1}: 置信度={box['confidence']:.3f}, "
                          f"边界框=[{box['bbox'][0]:.1f}, {box['bbox'][1]:.1f}, "
                          f"{box['bbox'][2]:.1f}, {box['bbox'][3]:.1f}]")

            # 可视化结果
            bee_detector.visualize_detections(detections)
        else:
            print("未在测试集中找到图片")


if __name__ == "__main__":
    main()