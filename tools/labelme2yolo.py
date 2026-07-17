"""
LabelMe格式(json)转YOLO格式(txt)工具
支持矩形、多边形、圆形、点等格式的自动转换
支持生成实例分割数据 (Polygon) 或 目标检测数据 (BBox)
"""

import json
import os
import argparse
import math
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import Counter
import hashlib
import csv

def save_rename_map(rename_map, map_file):
    """保存重命名映射表到 CSV (追加模式)"""
    file_exists = os.path.exists(map_file)
    
    with open(map_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Original_Path', 'New_Path', 'Original_Name', 'New_Name'])
        for original, new, orig_name, new_name in rename_map:
            writer.writerow([original, new, orig_name, new_name])

def get_short_name(filename, limit=250):
    """
    生成短文件名（保留扩展名，添加hash以防冲突）
    策略：保留前缀(30) + Hash(8) + 后缀
    """
    name_only, ext = os.path.splitext(filename)
    if len(filename) <= limit:
        return filename
    
    # 简单的缩短策略
    # 保留前50个字符 + MD5(全名)的前8位
    short_hash = hashlib.md5(name_only.encode('utf-8')).hexdigest()[:8]
    # 限制名字部分长度
    max_base_len = limit - len(ext) - 10 # 留余量
    if max_base_len < 50: 
        prefix = name_only[:max_base_len]
    else:
        prefix = name_only[:50]
        
    new_name = f"{prefix}_{short_hash}{ext}"
    return new_name

def shape2points(shape, height, width):
    """
    将LabelMe的shape转换为归一化的点列表
    参考自 lab2yolo.py
    """
    shape_type = shape['shape_type']
    points = shape['points']
    new_points = []

    if shape_type == 'polygon' or shape_type == 'linestrip':
        for p in points:
            x = np.clip(p[0], 0, width)
            y = np.clip(p[1], 0, height)
            new_points.append((x / width, y / height))
            
    elif shape_type == 'rectangle':
        (x1, y1), (x2, y2) = points
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])
        # 转换为4个顶点的多边形
        pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        for p in pts:
            x = np.clip(p[0], 0, width)
            y = np.clip(p[1], 0, height)
            new_points.append((x / width, y / height))
            
    elif shape_type == "circle":
        # 圆形转换为多边形（36个点）
        bearing_angles = list(range(0, 360, 10)) + [360]
        
        orig_x1 = points[0][0]
        orig_y1 = points[0][1]
        orig_x2 = points[1][0]
        orig_y2 = points[1][1]
        
        radius = math.sqrt((orig_x2 - orig_x1) ** 2 + (orig_y2 - orig_y1) ** 2)
        
        for i in range(len(bearing_angles) - 1):
            ad = math.radians(bearing_angles[i])
            x = orig_x1 + radius * math.cos(ad)
            y = orig_y1 + radius * math.sin(ad)
            
            x = np.clip(x, 0, width)
            y = np.clip(y, 0, height)
            new_points.append((x / width, y / height))
            
    elif shape_type == "point":
        # 点直接归一化
        p = points[0]
        x = np.clip(p[0], 0, width)
        y = np.clip(p[1], 0, height)
        new_points.append((x / width, y / height))
        
    return new_points

def points2bbox(points):
    """
    将归一化的点列表转换为 YOLO bbox 格式 (x_center, y_center, w, h)
    """
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    
    xmin = min(xs)
    xmax = max(xs)
    ymin = min(ys)
    ymax = max(ys)
    
    w = xmax - xmin
    h = ymax - ymin
    x_c = xmin + w / 2
    y_c = ymin + h / 2
    
    return x_c, y_c, w, h

class Labelme2YOLO:
    def __init__(self, class_map, task='segment', label_mapper=None):
        self.class_map = class_map
        self.task = task
        self.label_mapper = label_mapper  # Mapping dictionary: source_label -> target_label
        self.stats = Counter()
        self.unknown_labels = set()
        self.ignored_labels = set() # Track ignored labels (e.g., '屏蔽')
        self.label_counts = Counter()

    def convert(self, json_path, save_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            self.stats['load_error'] += 1
            return False

        height = data.get('imageHeight')
        width = data.get('imageWidth')
        
        if not height or not width:
             self.stats['missing_size'] += 1
             return False

        if not data.get('shapes'):
            self.stats['empty_shapes'] += 1
            # 生成空文件以保持对应关系
            with open(save_path, 'w', encoding='utf-8') as f:
                pass
            return True

        lines = []
        has_valid_label = False
        
        for shape in data['shapes']:
            label = shape['label']
            
            # Apply mapping if mapper exists
            if self.label_mapper:
                if label in self.label_mapper:
                    target_label = self.label_mapper[label]
                    # Check for ignore keyword "屏蔽"
                    if target_label == "屏蔽":
                        self.ignored_labels.add(label)
                        continue
                    label = target_label # Update label to target
                else:
                    # If mapper exists but label not found, assume ignore or keep? 
                    # User: "map to corresponding... if not specified, default to json"
                    # Wait, user said: "若未指定标签文件...不用映射". 
                    # But if label file IS specified, we should probably ignore unknown mappings or keep them?
                    # Usually mapping file is exhaustive. Let's treat as unknown/ignore if strict.
                    # However, to be safe, let's keep original if not in map? 
                    # User: "如果我给的是label.ini这种标签文件，那么需要把json中的标注映射到对应的标签名称上"
                    # Implicitly, if not in file, it might be an issue. 
                    # Let's count as unknown for now and skip to be safe, or keep?
                    # Let's assume if label_mapper is provided, ONLY mapped labels are valid (plus non-shield).
                    # But wait, what if the user only maps SOME labels?
                    # Let's assume strict mapping.
                    self.unknown_labels.add(label)
                    continue

            if label not in self.class_map:
                self.unknown_labels.add(label)
                continue
                
            has_valid_label = True
            class_id = self.class_map[label]
            points = shape2points(shape, height, width)
            
            if not points:
                continue

            if self.task == 'detect':
                x_c, y_c, w, h = points2bbox(points)
                line = f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}"
            else:
                coords = " ".join([f"{p[0]:.6f} {p[1]:.6f}" for p in points])
                line = f"{class_id} {coords}"
            
            lines.append(line)
            self.label_counts[label] += 1 # Count valid labels


        if not has_valid_label:
            self.stats['no_valid_labels'] += 1
        
        # 即使没有有效标签（可能被过滤了），也生成空文件
        with open(save_path, 'w', encoding='utf-8') as f:
            if lines:
                f.write("\n".join(lines))
                self.stats['converted'] += 1
            else:
                pass # 空文件
                
        return True

def load_classes(args):
    """
    Returns:
        class_map: dict {class_name: class_id}
        label_mapper: dict {source_label: target_label} or None
    """
    class_map = {}
    label_mapper = None
    
    # 1. Explicit classes list via command line
    if args.classes:
        for i, name in enumerate(args.classes):
            class_map[name] = i
        return class_map, None

    # 2. Class file provided
    if args.class_file and os.path.exists(args.class_file):
        with open(args.class_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        # Detect format: Mapping (2+ columns) vs List (1 column)
        is_mapping = False
        if lines:
            # Check first few lines to guess format
            for line in lines[:5]:
                parts = line.split()
                if len(parts) >= 2:
                    is_mapping = True
                    break
        
        if is_mapping:
            label_mapper = {}
            target_labels = set()
            
            for line in lines:
                parts = line.split()
                if len(parts) >= 2:
                    source = parts[0]
                    target = parts[-1] # Assume last column is target (handling spaces in source? No, usually split by whitespace implies source is first token)
                    # Example: "脱碳 	 	 	 屏蔽" -> parts[0]=脱碳, parts[-1]=屏蔽
                    
                    label_mapper[source] = target
                    if target != "屏蔽":
                        target_labels.add(target)
            
            # Generate class IDs for targets
            sorted_targets = sorted(list(target_labels))
            for i, name in enumerate(sorted_targets):
                class_map[name] = i
                
            print(f"Loaded Mapping Mode: {len(label_mapper)} mappings found.")
            print(f"Valid Target Classes ({len(class_map)}): {list(class_map.keys())}")
            
        else:
            # Simple list mode
            for i, name in enumerate(lines):
                class_map[name] = i
            print(f"Loaded List Mode: {len(class_map)} classes found.")

        return class_map, label_mapper

    # 3. No classes provided - Auto Discovery (Will be handled in main)
    return None, None

def scan_classes_from_jsons(json_dir, recursive=True):
    labels = set()

    print("Scanning JSONs for classes (Auto Discovery)...")
    count = 0
    for root, dirs, files in os.walk(json_dir):
        for fname in files:
            if not fname.lower().endswith('.json'):
                continue
            count += 1
            if count % 5000 == 0:
                print(f"  已扫描 {count} 个 JSON 文件...")
            try:
                with open(os.path.join(root, fname), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for shape in data.get('shapes', []):
                        labels.add(shape['label'])
            except:
                continue

    print(f"  共扫描 {count} 个 JSON 文件")
    sorted_labels = sorted(list(labels))
    class_map = {name: i for i, name in enumerate(sorted_labels)}
    print(f"Auto Discovery: Found {len(class_map)} unique labels.")
    return class_map


def get_image_extensions():
    return {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

def main():
    args = parse_args()
    
    # 确定保存路径逻辑
    # 如果 args.save_dir 为 None，则每个 json 转到其同级目录
    # 如果 args.save_dir 有值，则全部转到该目录，并检查同名冲突
    
    if args.save_dir and not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    # 加载类别
    class_map, label_mapper = load_classes(args)
    
    # 如果未指定类别，自动扫描
    if not class_map:
        if args.classes or (args.class_file and os.path.exists(args.class_file)):
             # File existed but empty or invalid?
             print("Warning: Class file provided but no classes found. Falling back to auto-discovery?")
        
        class_map = scan_classes_from_jsons(args.json_dir, args.recursive)
        
    if not class_map:
        print("Error: No classes found in JSONs and no class file specified.")
        return
    
    print(f"Final Class Map: {class_map}")
    print("-" * 50)
    
    converter = Labelme2YOLO(class_map, args.task, label_mapper)

    
    # 1. 扫描文件 (使用 os.walk 替代 glob，更高效)
    json_files = {}
    image_files = {}
    image_exts = get_image_extensions()

    print("扫描目录中...")
    scan_count = 0
    for root, dirs, files in os.walk(args.json_dir):
        for fname in files:
            scan_count += 1
            if scan_count % 10000 == 0:
                print(f"  已扫描 {scan_count} 个文件...")
            fpath = Path(root) / fname
            ext = fpath.suffix.lower()
            if ext == '.json':
                json_files[fpath.resolve()] = fpath.resolve()
            elif ext in image_exts:
                image_files[fpath.resolve()] = fpath.resolve()

    print(f"扫描完成，共 {scan_count} 个文件")
    
    # 建立映射以检查配对情况 (key: parent_dir/stem -> resolved_path)
    json_map = {Path(f).parent / Path(f).stem: f for f in json_files}
    image_map = {Path(f).parent / Path(f).stem: f for f in image_files}

    json_keys = set(json_map.keys())
    image_keys = set(image_map.keys())
    
    # 2. 统计
    # 有标注无图
    no_image_jsons = json_keys - image_keys
    # 有图无标注
    no_json_images = image_keys - json_keys
    # 有效配对
    valid_keys = json_keys & image_keys
    
    print(f"Found {len(json_files)} JSON files.")
    print(f"Found {len(image_files)} Image files.")
    print("-" * 50)

    # 3. 处理转换
    
    # 用于检测文件名冲突（仅当指定 save_dir 时有效）
    filename_counter = Counter()
    duplicate_files = []
    
    # 长文件名处理
    rename_count = 0
    rename_map = [] # [(orig_path, new_path, orig_name, new_name), ...]
    
    for json_file in tqdm(sorted(list(json_files)), desc="Converting"):
        # 确定保存路径
        
        current_filename = json_file.stem + '.txt'
        
        # 1. 检查文件名长度 (如果是默认同级目录，检查 json 文件名长度是否过长导致 txt 路径过长)
        # 这里主要检查生成的 txt 文件名是否需要缩短
        # 如果是同级目录，save_path 的目录部分很长，文件名可能还好。
        # 如果 save_dir 指定了，那主要看文件名。
        
        # 逻辑：如果 json 文件名本身就很长，生成的 txt 也会很长
        # 即使 save_dir 很短，文件名太长在 Windows 下也可能出问题 (总路径 > 260)
        # 这里简单起见，只检查文件名部分是否 > 250 (留点余量给路径)
        # 或者更严格：检查全路径？
        # 用户需求：若图像名称长度超过 256（通常指文件名），自动重命名。
        
        # 原始文件名 (对应图像名通常也是这个 stem)
        orig_stem = json_file.stem
        
        # 检查是否需要重命名
        if len(orig_stem) > 200: # 设定一个较安全的阈值，比如 200，留给扩展名和路径
            new_stem = get_short_name(orig_stem, limit=200) # 这里 limit 指 stem 的限制
            
            # 只有当名字确实变了才记录
            if new_stem != orig_stem:
                rename_count += 1
                # 记录映射
                # 注意：这里我们只重命名生成的 TXT 文件名，并不修改原始 JSON 或图片文件名
                # 除非... 用户希望把原始文件也重命名了？
                # "若图像的名称长度超过256，则自动重命名" -> 这通常意味着源文件也要改，否则转换后的 TXT 和图片对不上。
                # 但 labelme2yolo 是只读工具，通常不改源文件。
                # 如果只改输出的 TXT 名，那图片名没改，YOLO 训练时会找不到图片（因为要求 txt 和 jpg 同名）。
                # 因此：必须重命名输出的 TXT，并且**警告**用户图片也需要对应重命名，或者我们生成一份重命名脚本？
                # 或者：我们假设用户会用另一个工具处理图片，这里只处理 TXT 的生成名。
                # 最好的做法：输出的 TXT 使用短文件名，并且**在同级目录下生成一份改名后的图片副本**？不，太占空间。
                # 妥协方案：只生成短文件名的 TXT，并记录映射表。用户需要根据映射表去重命名图片。
                # 或者：直接告诉用户哪些文件过长。
                
                # 等等，如果我生成了 short.txt，但图片还是 long.jpg，YOLO 训练是不认的。
                # 用户说“自动重命名”，可能期望这个工具能把图片也顺便改了？
                # 但这个工具是 labelme2yolo，主要处理 json -> txt。
                # 如果我在这里改了 txt 名，图片没改，是没用的。
                # 让我们假设：输出的 TXT 使用短名，同时输出一份 CSV 告诉用户“请把这些图片也改名”。
                # 或者，更激进一点：直接重命名 JSON 和图片？风险太大。
                
                # 调整策略：仅在 save_dir 模式下，生成的 TXT 使用短名。
                # 并在报告中提示。
                # 为了“可解密”，我们保存 rename_map.csv
                
                rename_map.append((str(json_file), "Pending_Rename", orig_stem, new_stem))
                current_filename = new_stem + '.txt'
        
        if args.save_dir:
            # 指定了统一目录 -> 所有 txt 保存到 args.save_dir
            save_path = os.path.join(args.save_dir, current_filename)
            
            # 记录文件名以检测冲突
            filename_counter[current_filename] += 1
            if filename_counter[current_filename] > 1:
                duplicate_files.append(current_filename)
        else:
            # 未指定目录 -> 保存到 json 同级目录
            # 此时如果发生了重命名，save_path 的文件名部分会变
            save_path = str(json_file.parent / current_filename)
        
        converter.convert(str(json_file), save_path)

    # 为没有 JSON 的图片生成空标签文件
    if args.empty_label and no_json_images:
        empty_count = 0
        for key in tqdm(sorted(list(no_json_images)), desc="Empty labels"):
            image_file = image_map[key]

            if args.save_dir:
                save_path = str(Path(args.save_dir) / (image_file.stem + '.txt'))
            else:
                save_path = str(image_file.parent / (image_file.stem + '.txt'))

            # 避免覆盖已有的 txt
            if not os.path.exists(save_path):
                with open(save_path, 'w', encoding='utf-8') as f:
                    pass
                empty_count += 1

        print(f"  - 已为 {empty_count} 张无标注图片生成空标签文件")
            
    # 4. 输出统计报告
    print("\n" + "=" * 50)
    print("统计报告 (STATISTICS REPORT)")
    print("=" * 50)
    print(f"处理的 JSON 文件总数: {len(json_files)}")
    print(f"  - 转换成功: {converter.stats['converted']}")
    print(f"  - 空标注 (无shapes): {converter.stats['empty_shapes']}")
    print(f"  - 无有效标签 (被过滤): {converter.stats['no_valid_labels']}")
    print(f"  - 加载/格式错误: {converter.stats['load_error'] + converter.stats['missing_size']}")
    
    print("-" * 30)
    print(f"缺失配对检查:")
    print(f"  - 有图无标注 (No JSON): {len(no_json_images)}")
    print(f"  - 有标注无图 (No Image): {len(no_image_jsons)}")
    
    if rename_count > 0:
        print("-" * 30)
        print(f"⚠️  发现文件名过长 (>200字符) 并已重命名 TXT: {rename_count} 个")
        print(f"  注意：生成的 TXT 文件名已缩短，但原始图片文件名未修改。")
        print(f"  请务必根据生成的映射表重命名对应的图片，否则 YOLO 无法匹配！")
        
        # 保存映射表
        if args.save_dir:
            map_file = os.path.join(args.save_dir, 'rename_map.csv')
        else:
            map_file = os.path.join(args.json_dir, 'rename_map.csv')
            
        save_rename_map(rename_map, map_file)
        print(f"  重命名映射表已保存至: {map_file}")
        
    if args.save_dir and duplicate_files:
        print("-" * 30)
        print(f"⚠️  警告: 发现重名文件冲突 (Output Directory Conflict):")
        print(f"  指定了统一保存目录，但不同子目录下存在同名文件，导致互相覆盖。")
        print(f"  冲突文件数量: {len(duplicate_files)}")
        if len(duplicate_files) < 10:
             print(f"  冲突文件名: {', '.join(duplicate_files)}")
        else:
             print(f"  冲突文件名 (前10个): {', '.join(duplicate_files[:10])} ...")

    print("-" * 30)
    print(f"标签数量统计 (Label Counts):")
    for label, count in sorted(converter.label_counts.items()):
        print(f"  - {label}: {count}")

    if converter.unknown_labels:
        print("-" * 30)
        print(f"忽略的未知标签 ({len(converter.unknown_labels)} 类):")
        print(f"  {', '.join(sorted(list(converter.unknown_labels)))}")

    if converter.ignored_labels:
        print("-" * 30)
        print(f"已屏蔽的标签 ({len(converter.ignored_labels)} 类):")
        print(f"  {', '.join(sorted(list(converter.ignored_labels)))}")
        
    print("=" * 50)
    
    # Determine paths for data.yaml
    if args.save_dir:
        final_dir = args.save_dir
    else:
        final_dir = args.json_dir
        
    print(f"结果已保存至: {final_dir}")
    
    # 生成 data.yaml
    generate_data_yaml(final_dir, class_map)

def generate_data_yaml(output_dir, class_map):
    """
    生成 data.yaml 文件
    Args:
        output_dir: 保存目录 (也是 train/val 指向的目录)
        class_map: {name: id} 映射
    """
    yaml_path = os.path.join(output_dir, 'data.yaml')
    abs_dir = os.path.abspath(output_dir)
    
    content = []
    # 用户要求: 不要用path参数，只需要train和val以及类别
    content.append(f"train: {abs_dir}/images/train")
    content.append(f"val: {abs_dir}/images/val")
    content.append("")
    
    content.append(f"nc: {len(class_map)}")
    content.append("names:")
    
    # id -> name
    id_to_name = {v: k for k, v in class_map.items()}
    for i in sorted(id_to_name.keys()):
        content.append(f"  {i}: {id_to_name[i]}")
        
    try:
        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(content))
        print(f"data.yaml 已生成: {yaml_path}")
    except Exception as e:
        print(f"生成 data.yaml 失败: {e}")

def parse_args():
    parser = argparse.ArgumentParser(description="LabelMe JSON转YOLO TXT工具")
    
    parser.add_argument('--json-dir', type=str, default=r'Z:\14-调试数据\lxm\Dataset\极片类通用数据', 
                        help='JSON文件夹路径 (或包含JSON的根目录)')
    parser.add_argument('--save-dir', type=str, default=None, 
                        help='保存TXT的文件夹路径 (默认: 与JSON文件同目录)')
    
    parser.add_argument('--classes', type=str, nargs='+', 
                        help='类别名称列表 (按ID顺序)，例如: person car dog')
    parser.add_argument('--class-file', type=str, default=r"Z:\14-调试数据\lxm\Dataset\极片类通用数据\labels.ini", 
                        help='包含类别名称的文件路径 (支持映射格式：原标签 目标标签)')
    
    parser.add_argument('--task', type=str, choices=['segment', 'detect'], default='segment',
                        help='任务类型: segment (输出多边形) 或 detect (输出矩形框)')
    
    parser.add_argument('--recursive', action='store_false', 
                        help='是否递归查找子目录')

    parser.add_argument('--empty-label', action='store_false',
                        help='为没有JSON标注的图片生成空的TXT标签文件 (作为背景图参与训练)')

    return parser.parse_args()

if __name__ == '__main__':
    main()
