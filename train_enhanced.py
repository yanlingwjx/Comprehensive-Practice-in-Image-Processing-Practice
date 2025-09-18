from ultralytics import YOLO
import yaml

# 1. 加载自定义模型配置
model = YOLO('yolov8x_bee.yaml')

# 2. 开始训练
results = model.train(
    data=r'D:\PythonProject\yolo_dataset\enhanced_data.yaml',
    epochs=1,
    imgsz=480,           # 降低分辨率
    batch=2,             # 减小 batch
    name='bee_yolov8x_enhanced',
    pretrained=False,
    optimizer='AdamW',
    lr0=0.001,
    weight_decay=0.0005,
    warmup_epochs=3,
    patience=30,
    save_dir=r'D:\PythonProject\runs\detect\bees_x_enhanced',
    workers=0,           # Windows CPU 模式建议为 0
    amp=True             # 混合精度（CPU 支持有限，但可尝试）
)

# 3. 验证
results = model.val()
print(f"mAP50-95: {results.box.map50:.4f}")

# 4. 预测（使用 raw string）
results = model.predict(r'D:\PythonProject\yolo_dataset\images_enhanced\test\image001.jpg', conf=0.5)
print(f"检测到蜜蜂数量: {len(results[0].boxes)}")