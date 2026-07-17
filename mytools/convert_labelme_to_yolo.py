import json
import os
import random
import shutil
from glob import glob

from tqdm import tqdm


def convert_labelme_to_yolo(src_dir, dst_dir, val_split=0.2):
    # 1. 定义标签映射
    label_map = {"foil": 0, "焊点-无效": 1, "焊点-有效": 2, "焊点-边缘": 3}

    # 2. 创建目标目录结构
    for split in ["train", "val"]:
        os.makedirs(os.path.join(dst_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(dst_dir, "labels", split), exist_ok=True)

    # 3. 查找所有 json 文件
    json_files = glob(os.path.join(src_dir, "**/*.json"), recursive=True)
    print(f"找到 {len(json_files)} 个标注文件")

    # 4. 随机打乱并划分
    random.shuffle(json_files)
    val_count = int(len(json_files) * val_split)
    val_files = json_files[:val_count]
    train_files = json_files[val_count:]

    def process_files(files, split):
        print(f"处理 {split} 数据集...")
        for json_path in tqdm(files):
            try:
                with open(json_path, encoding="utf-8") as f:
                    data = json.load(f)

                img_w = data.get("imageWidth")
                img_h = data.get("imageHeight")

                # 如果 json 中没有宽高，尝试读取图片获取
                img_name = data.get("imagePath")
                json_dir = os.path.dirname(json_path)

                # 处理图片路径
                img_path = None
                if img_name:
                    img_path = os.path.join(json_dir, img_name)

                if not img_path or not os.path.exists(img_path):
                    # 尝试根据 json 文件名查找图片
                    base_name_no_ext = os.path.splitext(os.path.basename(json_path))[0]
                    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".BMP"]:
                        tmp_path = os.path.join(json_dir, base_name_no_ext + ext)
                        if os.path.exists(tmp_path):
                            img_path = tmp_path
                            break

                if not img_path or not os.path.exists(img_path):
                    print(f"\n警告: 找不到图片，对应 json: {json_path}")
                    continue

                # 如果 json 里没写宽高，从图片读
                if not img_w or not img_h:
                    from PIL import Image

                    with Image.open(img_path) as img:
                        img_w, img_h = img.size

                # 转换标注
                yolo_labels = []
                for shape in data["shapes"]:
                    label = shape["label"]
                    if label not in label_map:
                        # 尝试模糊匹配或者忽略
                        continue

                    class_idx = label_map[label]
                    points = shape["points"]

                    if not points or len(points) < 3:  # 至少需要3个点构成多边形
                        continue

                    # 归一化坐标
                    normalized_points = []
                    for p in points:
                        x = p[0] / img_w
                        y = p[1] / img_h
                        # 限制在 [0, 1] 范围内
                        x = max(0.0, min(1.0, x))
                        y = max(0.0, min(1.0, y))
                        normalized_points.extend([x, y])

                    # YOLO 实例分割格式: <class_idx> <x1> <y1> <x2> <y2> ...
                    label_line = f"{class_idx} " + " ".join([f"{p:.6f}" for p in normalized_points])
                    yolo_labels.append(label_line)

                if not yolo_labels:
                    # print(f"\n跳过无有效标注的文件: {json_path}")
                    continue

                # 为了防止重名，使用相对路径生成的唯一文件名
                rel_path = os.path.relpath(img_path, src_dir)
                safe_name = rel_path.replace(os.sep, "_").replace(" ", "_")

                # 复制图片
                dst_img_path = os.path.join(dst_dir, "images", split, safe_name)
                shutil.copy(img_path, dst_img_path)

                # 写入 label 文件
                label_file_name = os.path.splitext(safe_name)[0] + ".txt"
                dst_label_path = os.path.join(dst_dir, "labels", split, label_file_name)
                with open(dst_label_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(yolo_labels))
            except Exception as e:
                print(f"\n处理文件 {json_path} 时出错: {e}")

    process_files(train_files, "train")
    process_files(val_files, "val")

    # 5. 生成 data.yaml
    # 使用相对路径，这样数据集移动后也能用
    yaml_content = f"""path: {os.path.abspath(dst_dir)}
train: images/train
val: images/val

names:
  0: foil
  1: 焊点-无效
  2: 焊点-有效
  3: 焊点-边缘
"""
    with open(os.path.join(dst_dir, "data.yaml"), "w", encoding="utf-8") as f:
        f.write(yaml_content)
    print(f"\n转换完成！数据保存在 {dst_dir}")


if __name__ == "__main__":
    src = r"D:\Min\Projects\VSCodeProjects\dataset\instance_HD"
    dst = r"D:\Min\Projects\VSCodeProjects\dataset\instance_HD_yolo"

    # 检查源目录是否存在
    if not os.path.exists(src):
        print(f"错误: 源目录 {src} 不存在")
    else:
        convert_labelme_to_yolo(src, dst)
