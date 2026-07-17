"""
数据集统计分析工具
功能：
1. 递归遍历指定文件夹
2. 统计标注类型数量 (rectangle, polygon, circle, point 等)
3. 统计各类别标注数量
4. 统计图片文件格式数量 (jpg, png, bmp 等)
5. 统计 JSON 文件数量
6. 找出没有对应 JSON 标注的图片文件
"""

import os
import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from tqdm import tqdm

def analyze_dataset(root_dir):
    # 初始化统计数据
    stats = {
        'shape_types': Counter(),
        'class_counts': Counter(),
        'file_formats': Counter(),
        'json_count': 0,
        'image_count': 0,
        'missing_label_images': []
    }
    
    # 获取所有文件
    root_path = Path(root_dir)
    all_files = list(root_path.rglob('*'))
    
    # 过滤出文件（排除目录）
    files = [f for f in all_files if f.is_file()]
    
    # 图片扩展名集合
    img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
    
    # 建立文件索引（不含扩展名的路径 -> 扩展名列表）
    # 用于快速查找是否有对应的json
    file_map = defaultdict(set)
    
    print("正在扫描文件...")
    for f in tqdm(files):
        ext = f.suffix.lower()
        
        # 统计文件格式
        if ext in img_extensions:
            stats['file_formats'][ext] += 1
            stats['image_count'] += 1
            file_map[str(f.with_suffix(''))].add(ext)
            
        elif ext == '.json':
            stats['json_count'] += 1
            stats['file_formats'][ext] += 1
            file_map[str(f.with_suffix(''))].add(ext)
            
            # 解析 JSON 内容
            try:
                with open(f, 'r', encoding='utf-8') as json_file:
                    data = json.load(json_file)
                    
                if 'shapes' in data:
                    for shape in data['shapes']:
                        # 统计形状类型
                        shape_type = shape.get('shape_type', 'unknown')
                        stats['shape_types'][shape_type] += 1
                        
                        # 统计类别
                        label = shape.get('label', 'unknown')
                        stats['class_counts'][label] += 1
            except Exception as e:
                print(f"Error reading {f}: {e}")

    # 检查缺失标注的图片
    print("正在检查缺失标注...")
    for base_path, exts in file_map.items():
        # 如果有图片扩展名，但没有 .json
        has_image = any(ext in img_extensions for ext in exts)
        has_json = '.json' in exts
        
        if has_image and not has_json:
            # 找出具体的图片文件路径
            for ext in exts:
                if ext in img_extensions:
                    stats['missing_label_images'].append(f"{base_path}{ext}")

    return stats

def print_stats(stats):
    print("\n" + "="*50)
    print("数据集统计报告")
    print("="*50)
    
    print(f"\n[文件概览]")
    print(f"图片总数: {stats['image_count']}")
    print(f"JSON总数: {stats['json_count']}")
    
    print(f"\n[文件格式分布]")
    for fmt, count in stats['file_formats'].most_common():
        print(f"  {fmt}: {count}")
        
    print(f"\n[标注形状统计]")
    if stats['shape_types']:
        for shape, count in stats['shape_types'].most_common():
            print(f"  {shape}: {count}")
    else:
        print("  无标注形状数据")
        
    print(f"\n[类别数量统计]")
    if stats['class_counts']:
        for label, count in stats['class_counts'].most_common():
            print(f"  {label}: {count}")
    else:
        print("  无类别数据")
        
    print(f"\n[未标注图片] (共 {len(stats['missing_label_images'])} 张)")
    if stats['missing_label_images']:
        print("  前10个示例:")
        for img in stats['missing_label_images'][:10]:
            print(f"  {img}")
        if len(stats['missing_label_images']) > 10:
            print(f"  ... 以及其他 {len(stats['missing_label_images']) - 10} 张")
    else:
        print("  所有图片均有对应 JSON 文件")
        
    print("\n" + "="*50)

def main():
    parser = argparse.ArgumentParser(description="数据集统计分析工具")
    parser.add_argument('--dir', type=str, default=r'D:\Min\Projects\VSCodeProjects\dataset\instance_涂布缺陷\检测模型', help='要分析的根目录路径')
    parser.add_argument('--save-missing', type=str, default=None, help='将未标注图片列表保存到文件')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dir):
        print(f"错误: 目录不存在 - {args.dir}")
        return
        
    stats = analyze_dataset(args.dir)
    print_stats(stats)
    
    if args.save_missing and stats['missing_label_images']:
        try:
            with open(args.save_missing, 'w', encoding='utf-8') as f:
                for img in stats['missing_label_images']:
                    f.write(f"{img}\n")
            print(f"\n未标注图片列表已保存至: {args.save_missing}")
        except Exception as e:
            print(f"\n保存文件失败: {e}")

if __name__ == "__main__":
    main()
