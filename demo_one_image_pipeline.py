# demo_one_image_pipeline.py
import cv2
import random
from pathlib import Path
from enhance_dataset import (
    load_image, apply_smoothing, apply_sharpening,
    apply_edge_fusion, apply_brightness_contrast, apply_morphology
)

# --------------------------------------------------
# 1. 路径设置
# --------------------------------------------------
SAVE_ROOT = Path(r"D:\PythonProject\enhanced_single")  # 统一保存根目录
img_path  = Path(r"D:\PythonProject\ne.jpg")  # 任选一张图

SAVE_ROOT.mkdir(exist_ok=True)
img = load_image(img_path)
save_stem = img_path.stem  # 不含扩展名

# --------------------------------------------------
# 2. 固定随机种子
# --------------------------------------------------
random.seed(42)

# --------------------------------------------------
# 3. 保存函数
# --------------------------------------------------
def save_stage(name, im):
    f_name = f"{save_stem}_{name}.jpg"
    out_path = SAVE_ROOT / f_name
    cv2.imwrite(str(out_path), im)
    print(f"  已保存 → {out_path}")

# --------------------------------------------------
# 4. 依次处理并弹窗
# --------------------------------------------------
stages = [
    ("原始图像", lambda x: x),
    ("平滑滤波", apply_smoothing),
    ("锐化", apply_sharpening),
    ("边缘融合", apply_edge_fusion),
    ("亮度对比度", apply_brightness_contrast),
    ("形态学", apply_morphology),
]

cv2.namedWindow("stage", cv2.WINDOW_NORMAL)
for title, func in stages:
    print(f"\n>>> {title}")
    out = func(img.copy())
    save_stage(title, out)
    cv2.imshow("stage", out)
    cv2.setWindowTitle("stage", title)
    cv2.waitKey(0)

# --------------------------------------------------
# 5. 完整链最终效果
# --------------------------------------------------
print("\n>>> 完整增强链")
final = out  # 上一次已经是形态学，直接复用
save_stage("完整增强链", final)
cv2.imshow("stage", final)
cv2.setWindowTitle("stage", "完整增强链 —— 最终效果")
cv2.waitKey(0)
cv2.destroyAllWindows()