import argparse
import csv
import os

from tqdm import tqdm


def restore_filenames(map_file):
    """根据映射表恢复文件名."""
    if not os.path.exists(map_file):
        print(f"错误: 找不到映射文件 {map_file}")
        return

    print(f"正在读取映射表: {map_file}")

    restore_list = []

    try:
        with open(map_file, encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # 跳过表头
            # 检查表头格式
            # Original_Path,New_Path,Original_Name,New_Name

            for row in reader:
                if len(row) < 4:
                    continue
                # 我们只需要 New_Path (当前文件路径) 和 Original_Path (目标恢复路径)
                # 或者 New_Path 可能已经是绝对路径了？
                # check_long_paths 生成的是绝对路径吗？
                # 看之前的代码：display_path 和 new_display_path 是去掉了前缀的路径。
                # 但最好重新检查文件是否存在。

                # row[0]: Original_Path (这是我们想恢复成的样子)
                # row[1]: New_Path (这是文件现在的样子)

                orig_path = row[0]
                curr_path = row[1]

                restore_list.append((curr_path, orig_path))

    except Exception as e:
        print(f"读取 CSV 失败: {e}")
        return

    print(f"找到 {len(restore_list)} 条记录。准备恢复...")

    success_count = 0
    fail_count = 0
    not_found_count = 0

    # 倒序处理？防止目录依赖？这里只改文件名，应该没关系。
    # 但如果之前运行了多次，CSV 里可能有链式改名 A->B, B->C。
    # 这种情况下应该倒序恢复：先 C->B, 再 B->A。
    # 所以我们将列表翻转
    restore_list.reverse()

    for curr_path, orig_path in tqdm(restore_list, desc="恢复进度"):
        # 处理长路径前缀
        if os.name == "nt":
            if len(curr_path) > 250 and not curr_path.startswith("\\\\?\\"):
                curr_path = "\\\\?\\" + os.path.abspath(curr_path)
            if len(orig_path) > 250 and not orig_path.startswith("\\\\?\\"):
                orig_path = "\\\\?\\" + os.path.abspath(orig_path)

        if not os.path.exists(curr_path):
            # 尝试看看是不是已经恢复了？
            if os.path.exists(orig_path):
                # 已经存在目标文件，可能已经恢复过，或者未被修改
                # print(f"目标已存在，跳过: {orig_path}")
                continue

            # 真的找不到了
            # print(f"找不到当前文件: {curr_path}")
            not_found_count += 1
            continue

        try:
            os.rename(curr_path, orig_path)
            success_count += 1
        except OSError as e:
            print(f"恢复失败: {curr_path} -> {orig_path}")
            print(f"原因: {e}")
            fail_count += 1

    print("\n恢复完成报告:")
    print(f"  成功恢复: {success_count}")
    print(f"  失败: {fail_count}")
    print(f"  未找到文件 (可能已恢复): {not_found_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="根据映射表恢复文件名")
    parser.add_argument("--map", type=str, required=True, help="映射表 CSV 文件路径")
    args = parser.parse_args()

    restore_filenames(args.map)
