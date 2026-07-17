"""
修复 YOLO 标签文件：将纯 bbox 格式（无 polygon）转为带 polygon 的分割格式.

问题背景：
YOLO 分割训练在 verify_image_label 中，如果某个 label 文件所有行都没有 polygon
（每行 <= 6 个 token，即 class + cx,cy,w,h），则该文件的 segments 保持为空。
这导致数据集总 segments 计数与总 bboxes 计数不一致，触发 get_labels 中
"全部 segments 清空"的逻辑，最终训练崩溃于 cls_tensor 大小为 0。

本脚本将纯 bbox 行扩展为 4 角点 polygon：
  class cx cy w h  →  class x1 y1 x2 y2 x3 y3 x4 y4

用法:
  # 只检查，不修改
  python mytools/fix_bbox_to_polygon.py datasets/五里洋_yolo_dataset --dry-run

  # 正式修复
  python mytools/fix_bbox_to_polygon.py datasets/五里洋_yolo_dataset
"""

import argparse
import shutil
from pathlib import Path


def bbox_to_polygon(line: str):
    """将 bbox 行 (class cx cy w h) 转为带 4 角点 polygon 的行."""
    parts = line.strip().split()
    if len(parts) < 5:
        return None  # 格式异常

    cls = parts[0]
    cx, cy, w, h = map(float, parts[1:5])

    # bbox 四个角点：左上、右上、右下、左下
    x1, y1 = cx - w / 2, cy - h / 2
    x2, y2 = cx + w / 2, cy - h / 2
    x3, y3 = cx + w / 2, cy + h / 2
    x4, y4 = cx - w / 2, cy + h / 2

    return f"{cls} {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {x3:.6f} {y3:.6f} {x4:.6f} {y4:.6f}"


def is_bbox_only_line(line: str):
    """判断是否为纯 bbox 行（3~5 个 token，无 polygon）."""
    parts = line.strip().split()
    return 3 <= len(parts) <= 5


def has_polygon_line(lines):
    """判断列表中是否至少有一行包含 polygon（>6 个 token）."""
    for line in lines:
        if len(line.strip().split()) > 6:
            return True
    return False


def fix_label_dir(label_dir: str, backup: bool = True, fix_mixed: bool = False):
    """修复一个 label 目录下的所有 .txt 文件.

    Args:
        label_dir: label 子目录路径
        backup: 是否备份原始文件（后缀 .bak）
        fix_mixed: 是否也修复混合文件中的 bbox-only 行
    """
    label_path = Path(label_dir)
    if not label_path.exists():
        print(f"  目录不存在，跳过: {label_dir}")
        return

    txt_files = sorted(label_path.glob("*.txt"))
    print(f"  找到 {len(txt_files)} 个 label 文件")

    pure_bbox_files = 0
    total_fixed_lines = 0

    for txt_file in txt_files:
        with open(txt_file, encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        if not lines:
            continue

        fixed_lines = []
        needs_fix = False

        if not has_polygon_line(lines):
            # 整个文件全是纯 bbox → 每行都转为 polygon
            for line in lines:
                result = bbox_to_polygon(line)
                if result:
                    fixed_lines.append(result)
            if len(fixed_lines) == len(lines):
                needs_fix = True
                pure_bbox_files += 1
                total_fixed_lines += len(fixed_lines)
        elif fix_mixed:
            # 混合文件：只修复其中 bbox-only 的行
            mixed_count = 0
            for line in lines:
                if is_bbox_only_line(line):
                    result = bbox_to_polygon(line)
                    if result:
                        fixed_lines.append(result)
                        mixed_count += 1
                else:
                    fixed_lines.append(line)
            if mixed_count > 0:
                needs_fix = True
                total_fixed_lines += mixed_count

        if needs_fix:
            if backup:
                backup_path = txt_file.with_suffix(".txt.bak")
                shutil.copy2(txt_file, backup_path)

            with open(txt_file, "w", encoding="utf-8") as f:
                f.write("\n".join(fixed_lines) + "\n")

    print(f"  纯 bbox 文件（已全部转为 polygon）: {pure_bbox_files}")
    print(f"  修复行数: {total_fixed_lines}")


def delete_cache(label_dirs: list):
    """删除 label 缓存文件（*.cache），强制重建."""
    for d in label_dirs:
        label_path = Path(d)
        for pattern in ["train.cache", "val.cache", "test.cache", "labels.cache"]:
            cache_file = label_path / pattern
            if cache_file.exists():
                cache_file.unlink()
                print(f"  已删除缓存: {cache_file}")


def main():
    parser = argparse.ArgumentParser(description="修复 YOLO 分割数据集：将纯 bbox 标注转为带 polygon 的格式")
    parser.add_argument("dataset_dir", type=str, help="数据集根目录（包含 labels/train、labels/val 等子目录）")
    parser.add_argument("--no-backup", action="store_true", help="不备份原始文件")
    parser.add_argument(
        "--fix-mixed", action="store_true", help="也修复混合文件（有 polygon 也有 bbox）中的 bbox-only 行"
    )
    parser.add_argument("--no-clean-cache", action="store_true", help="不删除 label 缓存")
    parser.add_argument("--dry-run", action="store_true", help="只检查不修改")

    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    labels_dir = dataset_dir / "labels"

    if not labels_dir.exists():
        print(f"错误: labels 目录不存在: {labels_dir}")
        return

    # 查找所有 split 子目录（train / val / test）
    label_dirs = sorted([p for p in labels_dir.iterdir() if p.is_dir()])
    if not label_dirs:
        print(f"错误: labels 下没有子目录: {labels_dir}")
        return

    print(f"数据集目录: {dataset_dir}")
    print(f"Label 子目录: {[d.name for d in label_dirs]}")
    print(f"备份原文件: {'否' if args.no_backup else '是  (后缀 .bak)'}")
    print(f"修复混合文件: {'是' if args.fix_mixed else '否（仅处理全 bbox 文件）'}")
    print(f"模式: {'DRY RUN（只检查，不修改）' if args.dry_run else '正式修改'}")
    print()

    for label_dir in label_dirs:
        print(f"处理: {label_dir.name}/")
        if args.dry_run:
            count = 0
            for txt_file in label_dir.glob("*.txt"):
                with open(txt_file, encoding="utf-8") as f:
                    lines = [l.strip() for l in f if l.strip()]
                if lines and not has_polygon_line(lines):
                    count += 1
            print(f"  纯 bbox 文件（将被修复）: {count}")
        else:
            fix_label_dir(str(label_dir), backup=not args.no_backup, fix_mixed=args.fix_mixed)

    if not args.dry_run and not args.no_clean_cache:
        print("\n清理缓存...")
        delete_cache([str(d) for d in label_dirs])

    print("\n完成！" + (" (DRY RUN)" if args.dry_run else " 请重新运行训练。"))


if __name__ == "__main__":
    main()
