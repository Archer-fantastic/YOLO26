import argparse
import csv
import hashlib
import os

from tqdm import tqdm


def save_rename_map(rename_map, map_file):
    """保存重命名映射表到 CSV (追加模式)."""
    file_exists = os.path.exists(map_file)

    with open(map_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # 仅在文件不存在时写入表头
        if not file_exists:
            writer.writerow(["Original_Path", "New_Path", "Original_Name", "New_Name"])

        for original, new, orig_name, new_name in rename_map:
            writer.writerow([original, new, orig_name, new_name])


def check_long_paths(root_dir, limit=256, rename=False):
    """
    检查路径长度超过 limit 的文件
    :param root_dir: 根目录
    :param limit: 长度限制
    :param rename: 是否尝试自动重命名（截断文件名）.
    """
    print(f"正在检查路径长度超过 {limit} 字符的文件: {root_dir}")

    long_paths = []

    # 使用 os.walk 遍历，手动拼接路径
    # 注意：如果路径已经很长，os.walk 可能也会遇到问题，但在 Python 3.6+ 中通常能处理长路径（如果系统支持）
    # 为了保险，可以使用长路径前缀

    abs_root = os.path.abspath(root_dir)
    if os.name == "nt" and not abs_root.startswith("\\\\?\\"):
        search_root = "\\\\?\\" + abs_root
    else:
        search_root = abs_root

    for root, dirs, files in os.walk(search_root):
        for name in files:
            full_path = os.path.join(root, name)

            # 去掉前缀计算长度
            display_path = full_path
            if display_path.startswith("\\\\?\\"):
                display_path = display_path[4:]

            if len(display_path) >= limit:
                long_paths.append((full_path, display_path, len(display_path)))

    print(f"\n发现 {len(long_paths)} 个文件路径长度 >= {limit}")

    if not long_paths:
        return

    print("\n路径最长的 10 个文件:")
    # 按长度降序排序
    long_paths.sort(key=lambda x: x[2], reverse=True)

    for i, (full, display, length) in enumerate(long_paths[:10]):
        print(f"[{length}] {display}")

    if rename:
        print("\n正在重命名文件 (截断文件名)...")
        count = 0
        rename_map = []  # [(orig_path, new_path, orig_name, new_name), ...]

        # 预处理：按文件 stem 分组，确保同名不同后缀的文件被重命名为相同的新 stem
        # 1. 收集所有需要重命名的文件 (基于 stem)
        files_by_stem = {}  # { (dir_path, stem): [full_path1, full_path2, ...] }

        for full_path, display_path, length in long_paths:
            dirname = os.path.dirname(display_path)
            basename = os.path.basename(display_path)
            stem, ext = os.path.splitext(basename)
            key = (dirname, stem)

            if key not in files_by_stem:
                files_by_stem[key] = []
            files_by_stem[key].append(full_path)

        print(f"  发现 {len(files_by_stem)} 组同名文件需要重命名。")

        for (dirname, name_only), file_list in tqdm(files_by_stem.items(), desc="重命名进度"):
            # 计算新的短 stem (统一应用于该组所有文件)

            # 允许的最大 stem 长度 (保守估计，假设扩展名最长为10)
            # limit - len(dirname) - max_ext_len - 1 (separator) - safe_margin
            max_name_len = limit - len(dirname) - 10 - 1 - 5

            if max_name_len <= 1:
                print(f"跳过 {name_only}: 目录路径太长 ({len(dirname)}), 无法通过缩短文件名解决。")
                continue

            new_name_only = name_only[:max_name_len]
            # 添加 hash 避免重名 (基于原始 stem)
            short_hash = hashlib.md5(name_only.encode()).hexdigest()[:4]

            # 调整长度
            if len(new_name_only) > 5:
                new_name_only = new_name_only[:-5] + "_" + short_hash
            else:
                new_name_only = new_name_only + "_" + short_hash

            # 对该组下的每个文件应用新的 stem
            for full_path in file_list:
                # 获取该文件的实际扩展名
                # 注意：full_path 可能带前缀，用 os.path.splitext 处理
                _, ext = os.path.splitext(full_path)

                # 重新构建新文件名
                new_filename = new_name_only + ext
                new_full_path = os.path.join(os.path.dirname(full_path), new_filename)

                # 显示用的路径
                display_path = full_path
                if display_path.startswith("\\\\?\\"):
                    display_path = display_path[4:]

                new_display_path = new_full_path
                if new_display_path.startswith("\\\\?\\"):
                    new_display_path = new_display_path[4:]

                # 执行重命名
                try:
                    if os.path.exists(new_full_path):
                        # 如果目标已存在（可能是之前运行残留或hash碰撞），跳过或报错
                        # 这里简单跳过并警告
                        print(f"目标文件已存在，跳过: {new_filename}")
                        continue

                    os.rename(full_path, new_full_path)
                    count += 1
                    rename_map.append((display_path, new_display_path, os.path.basename(display_path), new_filename))
                except OSError as e:
                    print(f"重命名失败 {os.path.basename(display_path)}: {e}")

        print(f"成功重命名 {count} 个文件。")

        if count > 0:
            map_file = os.path.join(root_dir, "rename_map_long_paths.csv")
            save_rename_map(rename_map, map_file)
            print(f"重命名映射表已保存至: {map_file}")

    else:
        print("\n提示: 使用 --rename 参数可自动截断文件名 (实验性功能)。")


def parse_args():
    parser = argparse.ArgumentParser(description="检查并处理长文件路径")
    parser.add_argument(
        "--dir",
        type=str,
        default=r"D:\Min\Projects\VSCodeProjects\dataset\instance_涂布缺陷\检测模型",
        help="要扫描的目录路径",
    )
    parser.add_argument("--limit", type=int, default=256, help="路径长度限制 (默认: 256)")
    parser.add_argument("--rename", action="store_false", help="自动重命名以符合长度限制")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    check_long_paths(args.dir, args.limit, args.rename)
