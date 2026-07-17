"""
根据标注形状类型筛选数据集工具
功能：
1. 递归扫描指定目录下的 LabelMe JSON 文件
2. 检查 JSON 中是否包含指定的形状类型 (如 rectangle, polygon, circle, point)
3. 将符合条件的文件（JSON + 对应图片）复制或移动到新目录
"""

import os
import json
import argparse
import shutil
from pathlib import Path
from tqdm import tqdm

def filter_dataset(src_dir, dst_dir, target_shapes, mode='copy', recursive=True):
    """
    Args:
        src_dir: 源目录
        dst_dir: 目标目录
        target_shapes: 目标形状列表 (list of str)
        mode: 'copy' or 'move'
        recursive: 是否递归查找
    """
    # 确保目标目录存在
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
        
    src_path = Path(src_dir)
    
    # 查找所有 JSON 文件
    print(f"正在扫描 {src_dir} ...")
    if recursive:
        json_files = list(src_path.rglob('*.json'))
    else:
        json_files = list(src_path.glob('*.json'))
        
    print(f"找到 {len(json_files)} 个 JSON 文件")
    
    count = 0
    skipped = 0
    
    for json_file in tqdm(json_files, desc="Processing"):
        try:
            # 读取 JSON
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # 检查形状
            has_target_shape = False
            if 'shapes' in data:
                for shape in data['shapes']:
                    if shape['shape_type'] in target_shapes:
                        has_target_shape = True
                        break
            
            if has_target_shape:
                # 找到对应图片
                # 优先使用 json 中的 imagePath，但要注意路径可能是相对的或绝对的
                # 这里假设图片和 json 在同一目录下，且文件名一致（除了后缀）
                
                # 尝试推断图片路径
                img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
                img_file = None
                
                # 方法1: 同名文件查找
                for ext in img_extensions:
                    potential_img = json_file.with_suffix(ext)
                    if potential_img.exists():
                        img_file = potential_img
                        break
                        
                # 方法2: 如果同名文件不存在，尝试读取 json 中的 imagePath (处理相对路径)
                if not img_file:
                    img_name_in_json = data.get('imagePath', '')
                    if img_name_in_json:
                        potential_img = json_file.parent / img_name_in_json
                        if potential_img.exists():
                            img_file = potential_img

                if img_file:
                    # 执行操作
                    dst_json = os.path.join(dst_dir, json_file.name)
                    dst_img = os.path.join(dst_dir, img_file.name)
                    
                    # 处理重名
                    if os.path.exists(dst_json):
                        base_name = json_file.stem + f"_{count}"
                        dst_json = os.path.join(dst_dir, base_name + '.json')
                        dst_img = os.path.join(dst_dir, base_name + img_file.suffix)
                    
                    if mode == 'move':
                        shutil.move(str(json_file), dst_json)
                        shutil.move(str(img_file), dst_img)
                    else:
                        shutil.copy2(str(json_file), dst_json)
                        shutil.copy2(str(img_file), dst_img)
                        
                    count += 1
                else:
                    # print(f"Warning: Image not found for {json_file}")
                    skipped += 1
                    
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            
    action_name = "移动" if mode == 'move' else "复制"
    print(f"\n完成！已{action_name} {count} 组文件到 {dst_dir}")
    if skipped > 0:
        print(f"跳过 {skipped} 个文件（未找到对应图片）")

def main():
    parser = argparse.ArgumentParser(description="按标注形状筛选数据集工具")
    
    parser.add_argument('--src', type=str, default=r'', help='源数据集目录')
    parser.add_argument('--dst', type=str, default=r'', help='目标保存目录')
    
    parser.add_argument('--shape', type=str, default='rectangle', 
                        help='目标形状类型，支持多选 (e.g. rectangle polygon circle point)')
    
    parser.add_argument('--move', action='store_false', help='移动文件而不是复制 (默认复制)')
    parser.add_argument('--no-recursive', action='store_true', help='递归查找子目录')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.src):
        print(f"错误: 源目录不存在 - {args.src}")
        return
        
    mode = 'move' if args.move else 'copy'
    recursive = not args.no_recursive
    
    print(f"配置信息:")
    print(f"  源目录: {args.src}")
    print(f"  目标目录: {args.dst}")
    print(f"  筛选形状: {args.shape}")
    print(f"  操作模式: {mode}")
    print(f"  递归查找: {recursive}")
    
    filter_dataset(args.src, args.dst, args.shape, mode, recursive)

if __name__ == "__main__":
    main()
