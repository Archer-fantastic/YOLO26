"""
LabelMe JSON 转换为 YOLO Pose 格式
用于关键点检测任务

YOLO Pose 标签格式 (单类, 所有关键点在一行):
  0 x_center y_center width height kpt1_x kpt1_y kpt1_v kpt2_x kpt2_y kpt2_v ...
  (共 5 + num_keypoints * 3 列)
"""

import os
import json
import shutil
import random
from pathlib import Path
from tqdm import tqdm


def load_label_mapping(mapping_file):
    """
    从 label.ini 读取关键点映射

    label.ini 格式 (每行: 源标签 目标标签):
        A1 A1        → A1 是关键点0
        A2 A2        → A2 是关键点1
        B1 屏蔽       → 过滤掉 B1 标签
    """
    keypoint_map = {}
    disabled = set()
    kpt_id = 0

    with open(mapping_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            source, target = parts[0], parts[1]

            if target.lower() in ('屏蔽', 'ignore', 'none'):
                disabled.add(source)
                print(f"  屏蔽: {source}")
                continue

            keypoint_map[source] = kpt_id
            kpt_id += 1

    print(f"  关键点: {keypoint_map}")
    print(f"  屏蔽标签: {disabled}")
    return keypoint_map


class Labelme2PoseYOLO:
    def __init__(self, src_dir, dst_dir, keypoint_map=None, val_split=0.2):
        self.src_dir = Path(src_dir)
        self.dst_dir = Path(dst_dir)
        self.val_split = val_split
        self.keypoint_map = keypoint_map or {}
        self.num_keypoints = len(self.keypoint_map)
        self.stats = {'total': 0, 'converted': 0, 'skipped': 0, 'invalid': 0}

    def create_dirs(self):
        for split in ['train', 'val']:
            (self.dst_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
            (self.dst_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)

    def find_pairs(self):
        json_files = list(self.src_dir.rglob('*.json'))
        pairs = []

        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError):
                self.stats['invalid'] += 1
                continue

            image_name = data.get('imagePath', '')
            if not image_name:
                image_name = json_file.stem + '.bmp'

            image_path = json_file.parent / image_name
            if not image_path.exists():
                for ext in ['.bmp', '.jpg', '.png', '.jpeg']:
                    image_path = json_file.parent / (json_file.stem + ext)
                    if image_path.exists():
                        break

            if image_path.exists():
                pairs.append((json_file, image_path))
            else:
                self.stats['invalid'] += 1

        return pairs

    def convert_single(self, json_file, image_file):
        """
        转换单个标注文件，所有关键点合并为一行 (class_id=0)
        返回标签字符串，无有效标注返回 None
        """
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            img_w = data.get('imageWidth', 640)
            img_h = data.get('imageHeight', 640)
            shapes = data.get('shapes', [])

            if not shapes:
                return None

            keypoints = [0.0] * (self.num_keypoints * 3)
            kpt_xy_list = []
            has_valid = False

            for shape in shapes:
                if shape.get('shape_type', '') != 'point':
                    continue

                label = shape.get('label')
                if label not in self.keypoint_map:
                    continue

                points = shape.get('points', [])
                if not points or len(points[0]) < 2:
                    continue

                x, y = points[0][0], points[0][1]
                norm_x = x / img_w
                norm_y = y / img_h

                kpt_id = self.keypoint_map[label]
                keypoints[kpt_id * 3] = norm_x
                keypoints[kpt_id * 3 + 1] = norm_y
                keypoints[kpt_id * 3 + 2] = 2.0  # 可见
                kpt_xy_list.append((norm_x, norm_y))
                has_valid = True

            if not has_valid:
                return None

            # 计算 bbox (基于关键点外接框 + padding)
            xs = [p[0] for p in kpt_xy_list]
            ys = [p[1] for p in kpt_xy_list]
            cx = (min(xs) + max(xs)) / 2.0
            cy = (min(ys) + max(ys)) / 2.0
            bw = max(xs) - min(xs)
            bh = max(ys) - min(ys)

            # bbox 最小边长为图像的 5%，防止框过窄导致检测头无法定位
            min_size = 0.05
            if bw < min_size:
                half = (min_size - bw) / 2.0
                bw = min_size
                cx = max(bw / 2.0, min(1.0 - bw / 2.0, cx))
            if bh < min_size:
                half = (min_size - bh) / 2.0
                bh = min_size
                cy = max(bh / 2.0, min(1.0 - bh / 2.0, cy))

            kpt_str = ' '.join(f'{v:.6f}' for v in keypoints)
            return f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {kpt_str}"

        except Exception as e:
            print(f"转换失败 {json_file}: {e}")
            return None

    def convert(self):
        print(f"开始转换...")
        print(f"源目录: {self.src_dir}")
        print(f"目标目录: {self.dst_dir}")
        print(f"关键点映射: {self.keypoint_map}")
        print(f"关键点数量: {self.num_keypoints}")
        print("-" * 50)

        self.create_dirs()

        pairs = self.find_pairs()
        self.stats['total'] = len(pairs)
        print(f"找到 {len(pairs)} 对配对文件")

        random.shuffle(pairs)
        val_count = int(len(pairs) * self.val_split)
        val_pairs = pairs[:val_count]
        train_pairs = pairs[val_count:]

        print(f"训练集: {len(train_pairs)} 个")
        print(f"验证集: {len(val_pairs)} 个")
        print("-" * 50)

        self.process_split(train_pairs, 'train')
        self.process_split(val_pairs, 'val')
        self.generate_yaml()
        self.print_stats()

    def process_split(self, pairs, split):
        print(f"\n处理 {split} 数据集...")

        for json_file, image_file in tqdm(pairs, desc=split):
            label_line = self.convert_single(json_file, image_file)

            if label_line is None:
                self.stats['skipped'] += 1
                label_line = ""

            rel_path = image_file.relative_to(self.src_dir)
            safe_name = str(rel_path).replace('\\', '_').replace('/', '_')

            dst_img = self.dst_dir / 'images' / split / safe_name
            shutil.copy(image_file, dst_img)

            dst_label = self.dst_dir / 'labels' / split / (Path(safe_name).stem + '.txt')
            with open(dst_label, 'w', encoding='utf-8') as f:
                f.write(label_line)
                if label_line:
                    f.write('\n')

            self.stats['converted'] += 1

    def generate_yaml(self):
        yaml_path = self.dst_dir / 'data.yaml'

        id_to_label = {v: k for k, v in self.keypoint_map.items()}
        kpt_names = [id_to_label.get(i, f'kpt_{i}') for i in range(self.num_keypoints)]

        yaml_content = f"""# YOLO Pose 数据集配置文件
# 自动生成

path: {self.dst_dir.absolute().as_posix()}
train: images/train
val: images/val
test:

# 关键点配置
kpt_shape: [{self.num_keypoints}, 3]
flip_idx: {list(range(self.num_keypoints))}

# 类别
names:
  0: target

# 关键点名称
kpt_names:
"""
        for name in kpt_names:
            yaml_content += f"  - {name}\n"

        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write(yaml_content)

        print(f"\n配置文件已生成: {yaml_path}")

    def print_stats(self):
        print("\n" + "=" * 50)
        print("转换统计")
        print("=" * 50)
        print(f"总文件数: {self.stats['total']}")
        print(f"转换成功: {self.stats['converted']}")
        print(f"跳过(无标注): {self.stats['skipped']}")
        print(f"无效(找不到图像): {self.stats['invalid']}")
        print("=" * 50)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='LabelMe JSON -> YOLO Pose 格式转换')
    parser.add_argument('--src', type=str,
                        default=r"Z:\14-调试数据\lxm\Dataset\pose_涂布抓边\抓边模型\抓边模型\原始数据",
                        help='源数据目录')
    parser.add_argument('--dst', type=str,
                        default=r"Z:\14-调试数据\lxm\Dataset\pose_涂布抓边\抓边模型_yolo_data",
                        help='目标输出目录')
    parser.add_argument('--mapping-file', type=str, default=None,
                        help='label.ini 映射文件路径 (格式: 源标签 目标标签)')
    parser.add_argument('--val-split', type=float, default=0.2, help='验证集比例')
    args = parser.parse_args()

    if args.mapping_file and os.path.exists(args.mapping_file):
        print(f"读取映射文件: {args.mapping_file}")
        keypoint_map = load_label_mapping(args.mapping_file)
    else:
        keypoint_map = {'A1': 0, 'A2': 1}
        print("未指定映射文件, 使用默认映射")

    converter = Labelme2PoseYOLO(args.src, args.dst, keypoint_map=keypoint_map,
                                  val_split=args.val_split)
    converter.convert()

    print(f"\n转换完成!")
    print(f"数据集保存在: {args.dst}")
    print(f"\n训练命令:")
    print(f"python train/pose_train.py --data {args.dst}/data.yaml --weights yolo26n-pose.pt")


if __name__ == "__main__":
    main()
