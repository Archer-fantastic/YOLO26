import argparse
import os
import sys
from pathlib import Path

import cv2

# 强制从本项目加载 ultralytics (包含 Pose26 等自定义模块)
_project_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _project_root)

# 清除已缓存的 ultralytics 模块，确保从本地重新加载
_cached = [k for k in sys.modules if k.startswith("ultralytics")]
for _k in _cached:
    del sys.modules[_k]

from ultralytics import YOLO

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def detect_video(model, video_path, output_path, conf=0.3):
    """处理视频输入."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频 {video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"开始处理视频: {os.path.basename(video_path)} ({width}x{height}, {fps}fps)")
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=conf, verbose=False)
        annotated_frame = results[0].plot(boxes=True, labels=True, kpt_radius=3, kpt_line=True)
        out.write(annotated_frame)

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"  进度: {frame_count}/{total_frames} ({frame_count / max(total_frames, 1) * 100:.1f}%)")

    cap.release()
    out.release()
    print(f"视频处理完成: {output_path}")


def detect_image(model, image_path, output_path, conf=0.3):
    """处理单张图片."""
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"错误: 无法读取图片 {image_path}")
        return

    results = model.predict(frame, conf=conf, verbose=True)
    r = results[0]

    # 打印检测结果
    print(f"  检测到 {len(r.boxes)} 个目标, {len(r.keypoints)} 组关键点")
    if r.boxes is not None and len(r.boxes):
        for i, box in enumerate(r.boxes):
            print(f"    [{i}] class={int(box.cls)}, conf={box.conf.item():.3f}, bbox={box.xywh.tolist()}")
    if r.keypoints is not None and len(r.keypoints):
        print(f"    keypoints shape: {r.keypoints.data.shape}")

    annotated = r.plot(boxes=True, labels=True, kpt_radius=5, kpt_line=True, font_size=1.0)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, annotated)
    print(f"图片处理完成: {output_path}")


def detect_images(model, input_path, output_dir, conf=0.3):
    """处理图片目录."""
    image_files = []
    for p in Path(input_path).rglob("*"):
        if p.suffix.lower() in IMAGE_EXTS:
            image_files.append(p)

    if not image_files:
        print(f"错误: 未找到图片文件 {input_path}")
        return

    print(f"开始处理 {len(image_files)} 张图片")

    for i, img_path in enumerate(image_files):
        rel = img_path.relative_to(input_path)
        dst = Path(output_dir) / rel.with_suffix(".jpg")
        dst.parent.mkdir(parents=True, exist_ok=True)

        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        results = model.predict(frame, conf=conf, verbose=False)
        annotated = results[0].plot(boxes=True, labels=True, kpt_radius=3, kpt_line=True)
        cv2.imwrite(str(dst), annotated)

        if (i + 1) % 50 == 0 or (i + 1) == len(image_files):
            print(f"  进度: {i + 1}/{len(image_files)}")

    print(f"图片处理完成: {output_dir}")


def detect(model, input_path, output_path, conf=0.3):
    """
    自动识别输入类型 (视频/图片/目录) 并进行关键点检测.

    :param model: YOLO 模型
    :param input_path: 输入路径 (视频文件/图片文件/目录)
    :param output_path: 输出路径 (视频文件/图片文件/目录)
    :param conf: 置信度阈值
    """
    input_path = str(input_path)
    suffix = Path(input_path).suffix.lower()

    if os.path.isfile(input_path):
        if suffix in VIDEO_EXTS:
            detect_video(model, input_path, output_path, conf)
        elif suffix in IMAGE_EXTS:
            detect_image(model, input_path, output_path, conf)
        else:
            print(f"错误: 不支持的文件格式 '{suffix}'")
            print(f"  支持的视频: {', '.join(VIDEO_EXTS)}")
            print(f"  支持的图片: {', '.join(IMAGE_EXTS)}")
    elif os.path.isdir(input_path):
        detect_images(model, input_path, output_path, conf)
    else:
        print(f"错误: 路径不存在 {input_path}")


# ---------------------- 运行关键点检测 ----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO 关键点检测 (支持视频/图片/目录)")
    parser.add_argument("--input", type=str, required=True, help="输入路径 (视频/图片/目录)")
    parser.add_argument("--output", type=str, required=True, help="输出路径 (视频/图片/目录)")
    parser.add_argument("--weights", type=str, default="yolo26n-pose.pt", help="模型权重路径")
    parser.add_argument("--conf", type=float, default=0.3, help="置信度阈值")
    args = parser.parse_args()

    print(f"加载模型: {args.weights}")
    model = YOLO(args.weights)
    print(f"输入: {args.input}")
    print(f"输出: {args.output}")
    print("-" * 50)

    detect(model, args.input, args.output, conf=args.conf)
