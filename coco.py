# coco2yolo.py
import os
import json
from pathlib import Path
import shutil


def coco_to_yolo_bbox(bbox, img_width, img_height):
    """Convert COCO bbox [x,y,w,h] to YOLO format [x_center, y_center, w, h] (normalized)"""
    x, y, w, h = bbox
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    w_norm = w / img_width
    h_norm = h / img_height
    return [x_center, y_center, w_norm, h_norm]


def process_coco_annotations(json_path, images_src_dir, labels_dst_dir, image_dst_dir):
    """
    处理单个 COCO JSON 文件（如 train/_annotations.coco.json）
    images_src_dir: 包含图片和 JSON 的目录（或子目录）
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    images = {img['id']: img for img in data['images']}
    if 'annotations' not in data:
        print(f"❌ {json_path} 中缺少 'annotations' 字段")
        return

    annotations = data['annotations']
    ann_by_image = {}
    for ann in annotations:
        img_id = ann['image_id']
        if img_id not in ann_by_image:
            ann_by_image[img_id] = []
        ann_by_image[img_id].append(ann)

    # 类别映射（解决 Roboflow 多个 bees ID 的问题）
    cat_id_to_name = {cat['id']: cat['name'] for cat in data['categories']}
    unique_classes = sorted(set(cat_id_to_name.values()))
    print(f"   ├─ 检测到类别: {unique_classes}")
    print(f"   ├─ 图片数量: {len(images)}")
    print(f"   └─ 标注数量: {len(annotations)}")

    Path(labels_dst_dir).mkdir(parents=True, exist_ok=True)
    Path(image_dst_dir).mkdir(parents=True, exist_ok=True)

    processed = 0
    missing = 0

    for img_id, img_info in images.items():
        file_name = img_info['file_name']
        width, height = img_info['width'], img_info['height']

        # 尝试在 images_src_dir 中查找图片
        src_img_path = Path(images_src_dir) / file_name

        if not src_img_path.exists():
            # 常见情况：file_name 可能包含路径如 "train/filename.jpg"
            # 我们只取文件名部分
            filename_only = Path(file_name).name
            src_img_path = Path(images_src_dir) / filename_only

        if not src_img_path.exists():
            print(f"   ⚠️ 图片未找到: {file_name}")
            missing += 1
            continue

        # 复制图片到目标位置
        dst_img_path = Path(image_dst_dir) / src_img_path.name
        if not dst_img_path.exists():
            try:
                shutil.copy2(src_img_path, dst_img_path)
            except Exception as e:
                print(f"   ❌ 复制图片失败 {src_img_path} -> {dst_img_path}: {e}")
                continue

        # 写入 YOLO 标签文件
        label_path = Path(labels_dst_dir) / (src_img_path.stem + '.txt')
        with open(label_path, 'w') as f:
            if img_id in ann_by_image:
                for ann in ann_by_image[img_id]:
                    # 统一类别为 bee (class_id=0)
                    yolo_class_id = 0
                    bbox = ann['bbox']
                    yolo_bbox = coco_to_yolo_bbox(bbox, width, height)
                    f.write(f"{yolo_class_id} {' '.join(f'{x:.6f}' for x in yolo_bbox)}\n")

        processed += 1

    print(f"   ✅ 成功处理: {processed} 张图片 | 缺失: {missing}")


def convert_roboflow_to_yolo(roboflow_dir, output_dir):
    """
    主函数：转换 Roboflow 数据集到 YOLO 格式
    支持结构：
        train/
            image1.jpg
            image2.jpg
            _annotations.coco.json
        valid/
            ...
        test/
            ...
    """
    roboflow_dir = Path(roboflow_dir)
    output_dir = Path(output_dir)

    splits = ['train', 'valid', 'test']
    found_any = False

    for split in splits:
        print(f"\n🔄 正在处理 {split.upper()} 集...")
        split_dir = roboflow_dir / split
        annotations_json = split_dir / '_annotations.coco.json'

        if not annotations_json.exists():
            print(f"   ❌ 未找到标注文件: {annotations_json}")
            continue

        # ✅ 关键修改：图片就在 split_dir 目录下，无需 /images 子目录
        images_src = split_dir

        if not images_src.exists():
            print(f"   ❌ 未找到图片目录: {images_src}")
            continue

        # 输出路径保持不变
        image_dst = output_dir / 'images' / split
        label_dst = output_dir / 'labels' / split

        process_coco_annotations(
            json_path=annotations_json,
            images_src_dir=images_src,
            labels_dst_dir=label_dst,
            image_dst_dir=image_dst
        )
        found_any = True

    if not found_any:
        print("❌ 错误：未找到任何有效的数据集分割（train/valid/test）")
        return

    # 确保输出目录存在
    data_yaml = output_dir / 'data.yaml'
    data_yaml.parent.mkdir(parents=True, exist_ok=True)

    yaml_content = f"""train: {output_dir / 'images' / 'train'}
val: {output_dir / 'images' / 'valid'}
test: {output_dir / 'images' / 'test'}

# 类别数量
nc: 1

# 类别名称
names: ['bee']
"""
    data_yaml.write_text(yaml_content, encoding='utf-8')
    print(f"\n🎉 转换完成！YOLO 数据集已保存至：")
    print(f"   📁 {output_dir}")
    print(f"   📄 {data_yaml}")
    print(f"\n✅ 现在可以运行：")
    print(f"   yolo train data={data_yaml} model=yolov8n.pt epochs=100 imgsz=640")


# ======================
# ✅ 修改这里为您的路径
# ======================
if __name__ == "__main__":
    ROBOFLOW_DIR = r"E:\王进喜QQ\archive (1)"      # 您的数据集根目录
    OUTPUT_DIR   = r"D:\PythonProject\yolo_dataset"     # 输出目录（可自定义）

    convert_roboflow_to_yolo(ROBOFLOW_DIR, OUTPUT_DIR)