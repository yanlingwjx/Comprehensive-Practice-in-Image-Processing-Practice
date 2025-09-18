# enhance_dataset.py
import os
import cv2
import numpy as np
import random
from pathlib import Path
import shutil

def load_image(path):
    """加载图像"""
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"无法读取图像: {path}")
    return img

def save_image(img, save_path):
    """保存图像"""
    cv2.imwrite(str(save_path), img)

def apply_smoothing(img):
    """1️⃣ 平滑滤波：高斯模糊 + 双边滤波"""
    choice = random.random()
    if choice < 0.5:
        return cv2.GaussianBlur(img, (5, 5), 0)
    else:
        return cv2.bilateralFilter(img, 9, 75, 75)

def apply_sharpening(img):
    """2️⃣ 锐化：非锐化掩模"""
    gaussian = cv2.GaussianBlur(img, (9, 9), 10.0)
    return cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)

def apply_edge_fusion(img):
    """3️⃣ 边缘检测融合：Canny 边缘弱融合"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(img, 0.9, edges_color, 0.1, 0)

def apply_brightness_contrast(img):
    """4️⃣ 光照调整"""
    alpha = random.uniform(0.8, 1.2)  # 对比度
    beta = random.randint(-20, 20)     # 亮度
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def apply_morphology(img):
    """5️⃣ 形态学操作（模拟遮挡）"""
    if random.random() > 0.7:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        if random.random() > 0.5:
            img = cv2.dilate(img, kernel, iterations=1)
        else:
            img = cv2.erode(img, kernel, iterations=1)
    return img

def enhance_image(img):
    """组合多种图像处理方法"""
    # 随机选择是否增强
    if random.random() < 0.3:
        return img  # 保留原始图像

    # 按顺序或随机应用增强
    img = apply_smoothing(img)
    img = apply_sharpening(img)
    img = apply_edge_fusion(img)
    img = apply_brightness_contrast(img)
    img = apply_morphology(img)
    return np.clip(img, 0, 255).astype(np.uint8)

def enhance_dataset(
    src_images_dir: str,
    src_labels_dir: str,
    dst_images_dir: str,
    dst_labels_dir: str,
    max_images=None
):
    """
    离线增强图像数据集
    """
    src_images = Path(src_images_dir)
    src_labels = Path(src_labels_dir)
    dst_images = Path(dst_images_dir)
    dst_labels = Path(dst_labels_dir)

    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    image_files = list(src_images.glob("*.jpg")) + list(src_images.glob("*.png"))
    if max_images:
        image_files = image_files[:max_images]

    print(f"开始增强 {len(image_files)} 张图像...")

    for i, img_path in enumerate(image_files):
        try:
            img = load_image(img_path)
            enhanced_img = enhance_image(img.copy())

            # 保存增强图像
            save_path = dst_images / img_path.name
            save_image(enhanced_img, save_path)

            # 复制标签文件
            label_path = src_labels / f"{img_path.stem}.txt"
            if label_path.exists():
                shutil.copy(label_path, dst_labels / label_path.name)
            else:
                print(f"⚠️ 标签文件不存在: {label_path}")

            if i % 100 == 0:
                print(f"已完成 {i+1}/{len(image_files)}")

        except Exception as e:
            print(f"❌ 处理 {img_path.name} 失败: {e}")

    print(f"✅ 增强完成！增强图像保存至: {dst_images}")
    print(f"✅ 标签文件保存至: {dst_labels}")

# ==================== 运行增强 ====================
if __name__ == "__main__":
    # 修改为你的路径
    BASE_PATH = r"D:\PythonProject\yolo_dataset"

    # 增强训练集
    enhance_dataset(
        src_images_dir=fr"{BASE_PATH}\images\train",
        src_labels_dir=fr"{BASE_PATH}\labels\train",
        dst_images_dir=fr"{BASE_PATH}\images_enhanced\train",
        dst_labels_dir=fr"{BASE_PATH}\labels_enhanced\train",
        max_images=None  # 设为 500 调试
    )

    # 增强验证集（可选，建议只增强训练集）
    enhance_dataset(
        src_images_dir=fr"{BASE_PATH}\images\val",
        src_labels_dir=fr"{BASE_PATH}\labels\val",
        dst_images_dir=fr"{BASE_PATH}\images_enhanced\val",
        dst_labels_dir=fr"{BASE_PATH}\labels_enhanced\val",
        max_images=None
    )

    # 增强测试集（可选）
    enhance_dataset(
        src_images_dir=fr"{BASE_PATH}\images\test",
        src_labels_dir=fr"{BASE_PATH}\labels\test",
        dst_images_dir=fr"{BASE_PATH}\images_enhanced\test",
        dst_labels_dir=fr"{BASE_PATH}\labels_enhanced\test",
        max_images=None
    )