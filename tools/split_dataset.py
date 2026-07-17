"""
将数据集划分为训练集，验证集，测试集
支持命令行参数运行
"""
import os
import argparse
import random
import shutil
from pathlib import Path
from tqdm import tqdm
from collections import Counter

def makedir(new_dir):
    """创建目录"""
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

def count_labels(label_file):
    """读取标签文件并返回类别列表"""
    if not os.path.exists(label_file):
        return []
    
    classes = []
    try:
        with open(label_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    cls_id = line.split()[0]
                    classes.append(cls_id)
    except Exception:
        pass
    return classes

def print_distribution(counter, title="分布统计"):
    """打印标签分布"""
    print(f"\n--- {title} ---")
    if not counter:
        print("  (无数据)")
        return
    
    # 按数量降序排列
    sorted_counts = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    for cls, count in sorted_counts:
        print(f"  类别 {cls}: {count}")

def split_data(img_dir, label_dir, split_dir, split_ratio=[0.8, 0.2, 0.0], seed=42, link_mode='copy'):
    '''
    Args:
        img_dir: 原图片数据集路径
        label_dir: 原yolo格式txt文件数据集路径
        split_dir: 划分后数据集保存路径
        split_ratio: 划分比例 [train, val, test]
        seed: 随机种子
        link_mode: 'copy' (复制), 'symlink' (软链接), 'hardlink' (硬链接)
    '''
    random.seed(seed)
    
    # ... (原有代码保持不变直到复制部分) ...
    # 为了简化修改，我们重新定义 copy_file 函数并替换原有逻辑
    
    # 确保路径存在
    if not os.path.exists(img_dir):
        print(f"Error: Image directory not found: {img_dir}")
        return
    
    if not os.path.exists(label_dir):
        print(f"Warning: Label directory not found: {label_dir}. Treating all images as background (empty labels).")

    # 确定输出目录结构
    images_dir = os.path.join(split_dir, 'images')
    labels_dir = os.path.join(split_dir, 'labels')
    
    # 判断是否需要测试集
    has_test = len(split_ratio) == 3 and split_ratio[2] > 0
    subsets = ['train', 'val']
    if has_test:
        subsets.append('test')

    for root in [images_dir, labels_dir]:
        for subset in subsets:
            makedir(os.path.join(root, subset))

    # 计算划分点
    total_ratio = sum(split_ratio)
    if total_ratio != 1.0:
        split_ratio = [r / total_ratio for r in split_ratio]
    
    train_pct = split_ratio[0]
    
    # 获取图片列表
    images = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    random.shuffle(images)
    
    image_count = len(images)
    train_point = int(image_count * train_pct)
    valid_point = int(image_count * (train_pct + split_ratio[1]))

    print(f"图片总数: {image_count}")
    print(f"划分计划 -> 训练集: {train_point}, 验证集: {valid_point - train_point}, 测试集: {image_count - valid_point}")
    
    # 模式显示中文映射
    mode_map = {
        'copy': '复制文件 (Copy)',
        'symlink': '软链接 (Symlink)',
        'hardlink': '硬链接 (Hardlink)'
    }
    print(f"传输模式: {mode_map.get(link_mode, link_mode)}")

    # 统计计数器
    stats = {
        'total': Counter(),
        'train': Counter(),
        'val': Counter(),
        'test': Counter()
    }
    
    empty_label_count = 0

    def transfer_file(src, dst, mode):
        if mode == 'copy':
            shutil.copy2(src, dst)
        elif mode == 'symlink':
            # Windows下需要管理员权限或启用开发者模式才能创建符号链接
            # 如果失败，回退到 copy? 或者报错
            try:
                os.symlink(os.path.abspath(src), dst)
            except OSError as e:
                # Windows Error 1314: A required privilege is not held by the client
                if hasattr(e, 'winerror') and e.winerror == 1314:
                    print(f"Error: Creating symlink failed (Permission denied). Try running as Administrator or use --link-mode copy.")
                    raise e
                else:
                    raise e
        elif mode == 'hardlink':
            # 硬链接不需要管理员权限，但不能跨分区
            try:
                os.link(src, dst)
            except OSError:
                 # 跨分区会失败
                 print(f"Error: Hard link failed (Cross-device link?). Falling back to copy for {src}")
                 shutil.copy2(src, dst)

    for i in tqdm(range(image_count), desc=f'正在处理 {os.path.basename(img_dir)}'):
        # ... (原有逻辑)
        if i < train_point:
            subset = 'train'
        elif i < valid_point:
            subset = 'val'
        else:
            subset = 'test'
            
        out_img_dir = os.path.join(images_dir, subset)
        out_label_dir = os.path.join(labels_dir, subset)
        
        img_name = images[i]
        basename = os.path.splitext(img_name)[0]
        label_name = basename + '.txt'
        
        src_img_path = os.path.join(img_dir, img_name)
        if label_dir and os.path.exists(label_dir):
            src_label_path = os.path.join(label_dir, label_name)
        else:
            src_label_path = None
        
        dst_img_path = os.path.join(out_img_dir, img_name)
        dst_label_path = os.path.join(out_label_dir, label_name)
        
        # 处理同名冲突 (略微简化，假设已经处理好或者用之前的逻辑)
        if os.path.exists(dst_img_path):
            new_basename = f"{basename}_new"
            dst_img_path = os.path.join(out_img_dir, f"{new_basename}{os.path.splitext(img_name)[1]}")
            dst_label_path = os.path.join(out_label_dir, f"{new_basename}.txt")

        # 传输图片
        try:
            transfer_file(src_img_path, dst_img_path, link_mode)
        except Exception as e:
             # 如果链接失败，可能就中断了
             print(f"Transfer failed: {e}")
             continue
        
        # 处理标签
        labels = []
        if src_label_path and os.path.exists(src_label_path):
            # 标签文件很小，直接复制也没关系，但为了统一也支持链接
            # 不过如果要读取内容统计，还是得读
            labels = count_labels(src_label_path)
            transfer_file(src_label_path, dst_label_path, link_mode)
        else:
            with open(dst_label_path, 'w', encoding='utf-8') as f:
                pass
        
        if not labels:
            empty_label_count += 1
        
        for cls in labels:
            stats['total'][cls] += 1
            stats[subset][cls] += 1

    # ... (输出统计)
    print("\n" + "="*50)
    print("数据集划分统计报告")
    print("="*50)
    print(f"总图片数: {image_count}")
    print(f"空标注图片数 (背景图): {empty_label_count}")
    
    print_distribution(stats['total'], "原始数据集标签分布")
    print_distribution(stats['train'], "训练集标签分布")
    print_distribution(stats['val'], "验证集标签分布")
    if has_test:
        print_distribution(stats['test'], "测试集标签分布")
    print("="*50 + "\n")

def split_dataset_recursive(root_dir, save_dir, split_ratio, seed, link_mode='copy'):
    """递归查找目录并合并处理"""
    # 查找所有包含图片的子目录
    # 修改逻辑：不再依赖 _yolo_txt 目录结构
    # 而是收集所有图片，统一调用 split_data_list 进行处理
    
    print(f"正在递归扫描图片: {root_dir}")
    
    all_images = []
    # 常见图片扩展名
    img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
    
    for root, dirs, files in os.walk(root_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in img_exts:
                all_images.append(os.path.join(root, f))
    
    if not all_images:
        print(f"在 {root_dir} 中未找到任何图片。")
        return

    print(f"共发现 {len(all_images)} 张图片。")
    
    # 直接调用 split_data_list，它会自动处理标签匹配
    split_data_list(all_images, save_dir, split_ratio, seed, link_mode)

# 同时更新 split_data_pairs 和 split_data_list 支持 link_mode
def split_data_pairs(pairs, split_dir, split_ratio, seed, link_mode='copy'):
    # ... (逻辑同上，替换 shutil.copy2 为 transfer_file) ...
    # 由于篇幅限制，这里仅展示概念，实际代码需要完整替换
    pass 



def main():
    args = parse_args()
    
    # 检查比例参数
    if len(args.ratio) < 2 or len(args.ratio) > 3:
        print("Error: --ratio requires 2 or 3 numbers (e.g., 0.8 0.2)")
        return
    
    # 补全比例 (如果只有2个，补0作为test)
    split_ratio = args.ratio if len(args.ratio) == 3 else list(args.ratio) + [0.0]
    
    # 确保输出目录存在
    makedir(args.save_dir)

    # 推断 label_dir
    # 如果未指定 label_dir，默认认为 txt 文件和图片在同一个目录
    if args.label_dir is None:
        args.label_dir = args.img_dir
        print(f"Auto-detected label dir (same as img dir): {args.label_dir}")
    
    # 递归模式下，通常不需要指定 label_dir，因为是成对查找的
    # 但如果用户指定了 label_dir，在递归模式下其实意义不大，或者应该报错？
    # 现有的 split_dataset_recursive 逻辑是查找同级的 _yolo_txt 目录
    # 我们修改 recursive 逻辑，使其更通用：
    # 1. 如果指定了 recursive，则 img_dir 是根目录
    # 2. 遍历根目录下所有图片
    # 3. 对于每张图片，去寻找对应的 txt
    #    a. 在图片同级目录下找
    #    b. 在图片同级目录下的 _yolo_txt 目录找 (兼容旧逻辑)
    #    c. 在 args.label_dir (如果指定了) 下找？这在递归模式下很难对应
    
    if args.recursive:
        print(f"Mode: Recursive search in {args.img_dir}")
        # 新的递归逻辑：收集所有图片，统一处理
        # 这样可以解决同名文件问题（因为我们会重命名）
        
        all_images = []
        # 常见图片扩展名
        img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
        
        for root, dirs, files in os.walk(args.img_dir):
            for f in files:
                if os.path.splitext(f)[1].lower() in img_exts:
                    all_images.append(os.path.join(root, f))
        
        if not all_images:
            print(f"No images found in {args.img_dir}")
            return

        print(f"Found {len(all_images)} images recursively.")
        
        # 调用核心划分逻辑，传入图片列表而不是目录
        split_data_list(all_images, args.save_dir, split_ratio, args.seed, args.link_mode)
        
    else:
        print(f"Mode: Single directory split")
        # 获取目录下所有图片
        img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
        all_images = [os.path.join(args.img_dir, f) for f in os.listdir(args.img_dir) 
                      if os.path.splitext(f)[1].lower() in img_exts]
        
        if not all_images:
            print(f"No images found in {args.img_dir}")
            return
            
        # 如果指定了 label_dir，需要传递给处理函数以便查找
        # 但为了统一接口，我们可以在 split_data_list 内部处理
        # 或者我们预先构建 (img_path, label_path) 的列表
        
        # 为了兼容 split_data_list 的逻辑（它需要自己找 label），我们可以在这里封装一下
        # 但 split_data_list 最好设计成接收 (img_path, label_path) 元组列表
        
        # 构建配对列表
        image_label_pairs = []
        for img_path in all_images:
            basename = os.path.splitext(os.path.basename(img_path))[0]
            label_name = basename + '.txt'
            
            # 优先在 label_dir 找
            label_path = os.path.join(args.label_dir, label_name)
            if not os.path.exists(label_path):
                # 尝试在图片同级目录找 (如果 label_dir != img_dir)
                if args.label_dir != args.img_dir:
                    alt_path = os.path.join(args.img_dir, label_name)
                    if os.path.exists(alt_path):
                        label_path = alt_path
                    else:
                        label_path = None # 没找到
                else:
                    label_path = None
            
            image_label_pairs.append((img_path, label_path))
            
        split_data_pairs(image_label_pairs, args.save_dir, split_ratio, args.seed, args.link_mode)

def split_data_pairs(pairs, split_dir, split_ratio, seed, link_mode='copy'):
    """
    处理 (img_path, label_path) 配对列表
    """
    import uuid
    
    random.seed(seed)
    
    def transfer_file(src, dst, mode):
        # 统一处理 Windows 长路径前缀
        if os.name == 'nt':
            try:
                if len(os.path.abspath(src)) > 259 and not src.startswith("\\\\?\\"):
                    src = "\\\\?\\" + os.path.abspath(src)
                if len(os.path.abspath(dst)) > 259 and not dst.startswith("\\\\?\\"):
                    dst = "\\\\?\\" + os.path.abspath(dst)
            except Exception:
                pass

        try:
            if mode == 'copy':
                shutil.copy2(src, dst)
            elif mode == 'symlink':
                try:
                    os.symlink(os.path.abspath(src), dst)
                except OSError as e:
                    if hasattr(e, 'winerror') and e.winerror == 1314:
                        print(f"Warning: Symlink failed (Permission denied). Falling back to copy for {src}")
                        shutil.copy2(src, dst)
                    else:
                        print(f"Warning: Symlink failed ({e}). Falling back to copy for {src}")
                        shutil.copy2(src, dst)
            elif mode == 'hardlink':
                try:
                    os.link(src, dst)
                except OSError as e:
                     print(f"Warning: Hard link failed ({e}). Falling back to copy for {src}")
                     shutil.copy2(src, dst)
        except Exception as e:
            print(f"Error transferring file: {src} -> {dst}")
            print(f"Reason: {e}")
            # 不抛出异常，让循环继续

    random.shuffle(pairs)
    
    image_count = len(pairs)
    
    # 计算划分点
    total_ratio = sum(split_ratio)
    if total_ratio != 1.0:
        split_ratio = [r / total_ratio for r in split_ratio]
        
    train_point = int(image_count * split_ratio[0])
    valid_point = int(image_count * (split_ratio[0] + split_ratio[1]))
    
    # 目录准备
    images_dir = os.path.join(split_dir, 'images')
    labels_dir = os.path.join(split_dir, 'labels')
    
    has_test = len(split_ratio) == 3 and split_ratio[2] > 0
    subsets = ['train', 'val']
    if has_test:
        subsets.append('test')

    for root in [images_dir, labels_dir]:
        for subset in subsets:
            makedir(os.path.join(root, subset))
            
    # 统计
    stats = {
        'total': Counter(),
        'train': Counter(),
        'val': Counter(),
        'test': Counter()
    }
    empty_label_count = 0
    
    print(f"开始处理 {image_count} 张图片...")
    
    for i in tqdm(range(image_count)):
        img_path, label_path = pairs[i]
        
        if i < train_point:
            subset = 'train'
        elif i < valid_point:
            subset = 'val'
        else:
            subset = 'test'
            
        out_img_dir = os.path.join(images_dir, subset)
        out_label_dir = os.path.join(labels_dir, subset)
        
        # 处理文件名
        # 使用 UUID 防止重名，保留原扩展名
        # 或者保留原名，如果有冲突再加 UUID？
        # 用户通常喜欢保留原名。
        
        basename = os.path.basename(img_path)
        name_only, ext = os.path.splitext(basename)
        
        dst_img_path = os.path.join(out_img_dir, basename)
        dst_label_path = os.path.join(out_label_dir, name_only + '.txt')
        
        # 冲突检测与重命名
        if os.path.exists(dst_img_path):
            # 生成唯一后缀
            unique_suffix = str(uuid.uuid4())[:8]
            new_name = f"{name_only}_{unique_suffix}{ext}"
            new_label_name = f"{name_only}_{unique_suffix}.txt"
            
            dst_img_path = os.path.join(out_img_dir, new_name)
            dst_label_path = os.path.join(out_label_dir, new_label_name)
            
        # 复制图片
        transfer_file(img_path, dst_img_path, link_mode)
        
        # 处理标签
        labels = []
        if label_path and os.path.exists(label_path):
            transfer_file(label_path, dst_label_path, link_mode)
            labels = count_labels(label_path)
        else:
            # 创建空标签
            with open(dst_label_path, 'w', encoding='utf-8') as f:
                pass
        
        if not labels:
            empty_label_count += 1
            
        for cls in labels:
            stats['total'][cls] += 1
            stats[subset][cls] += 1
            
    # 输出报告
    print("\n" + "="*50)
    print("数据集划分统计报告")
    print("="*50)
    print(f"总图片数: {image_count}")
    print(f"空标注图片数 (背景图): {empty_label_count}")
    
    print_distribution(stats['total'], "原始数据集标签分布")
    print_distribution(stats['train'], "训练集标签分布")
    print_distribution(stats['val'], "验证集标签分布")
    if has_test:
        print_distribution(stats['test'], "测试集标签分布")
    print("="*50 + "\n")

def split_data_list(img_list, split_dir, split_ratio, seed, link_mode='copy'):
    """
    递归模式下的辅助函数：自动寻找标签并构建配对
    """
    pairs = []
    for img_path in img_list:
        # 寻找逻辑：
        # 1. 同级目录下的同名 .txt
        # 2. 同级目录下的 _yolo_txt 文件夹下的同名 .txt (兼容旧逻辑)
        
        parent = os.path.dirname(img_path)
        basename = os.path.splitext(os.path.basename(img_path))[0]
        txt_name = basename + '.txt'
        
        # 1. 同级
        p1 = os.path.join(parent, txt_name)
        if os.path.exists(p1):
            pairs.append((img_path, p1))
            continue
            
        # 2. _yolo_txt 子目录
        # 假设 _yolo_txt 是同级的，或者是在 parent 的兄弟目录？
        # 旧逻辑是：parent_yolo_txt
        # 这里尝试找 parent/../parent_yolo_txt/txt_name
        # 或者 parent/_yolo_txt/txt_name
        
        # 尝试方案 A: 当前目录下的 yolo_txt (例如 images/1.jpg, images/yolo_txt/1.txt)
        p2 = os.path.join(parent, 'yolo_txt', txt_name)
        if os.path.exists(p2):
            pairs.append((img_path, p2))
            continue
            
        # 尝试方案 B: 同级目录加后缀 (例如 data/images/1.jpg, data/images_yolo_txt/1.txt)
        parent_name = os.path.basename(parent)
        grandparent = os.path.dirname(parent)
        p3 = os.path.join(grandparent, f"{parent_name}_yolo_txt", txt_name)
        if os.path.exists(p3):
            pairs.append((img_path, p3))
            continue
            
        # 没找到标签 -> 空标签
        pairs.append((img_path, None))
        
    split_data_pairs(pairs, split_dir, split_ratio, seed, link_mode)


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO数据集划分工具")
    
    parser.add_argument('--img-dir', type=str, default=r'Z:\14-调试数据\lxm\Dataset\极片类通用数据', 
                        help='原图片数据集路径 (如果是递归模式，则为根目录)')
    parser.add_argument('--save-dir', type=str, default=r'Z:\14-调试数据\lxm\Dataset\极片类通用数据\yolo_dataset', 
                        help='划分后数据集保存路径')
    parser.add_argument('--label-dir', type=str, default=None, 
                        help='原label数据集路径 (非递归模式下，默认为 img_dir，即与图片同级)')
    
    parser.add_argument('--ratio', type=float, nargs='+', default=[0.7, 0.3, 0.1], 
                        help='划分比例，支持 2个(train val) 或 3个(train val test) 参数。例如: 0.8 0.2 或 0.7 0.2 0.1')
    
    parser.add_argument('--recursive', action='store_false', 
                        help='是否递归查找子目录')
    
    parser.add_argument('--seed', type=int, default=42, help='随机种子')

    parser.add_argument('--link-mode', type=str, choices=['copy', 'symlink', 'hardlink'], default='copy',
                        help='文件传输模式: copy (默认, 复制文件), symlink (软链接, 节省空间但需管理员权限), hardlink (硬链接, 节省空间且无需管理员权限，但不能跨分区)')
    
    return parser.parse_args()
if __name__ == "__main__":
    main()
