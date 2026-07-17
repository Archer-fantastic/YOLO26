from ultralytics import YOLO
import os
import argparse
import cv2
import numpy as np
import json
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from glob import glob

class YOLODetectorPredictor:
    def __init__(self, model_path, draw_chinese=False):
        self.model = YOLO(model_path)
        self.class_names = self.model.names
        self.draw_chinese = draw_chinese
        print(f"已加载目标检测模型：{model_path}，包含类别：{list(self.class_names.values())}")
        print(f"中文标签绘制：{'开启' if draw_chinese else '关闭'}")
        self.font = self._load_chinese_font() if draw_chinese else None

    def _load_chinese_font(self):
        font_paths = [
            "C:/Windows/Fonts/simhei.ttf",
            "C:/Windows/Fonts/msyh.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/Library/Fonts/PingFang.ttc"
        ]
        for path in font_paths:
            if os.path.exists(path):
                return ImageFont.truetype(path, 30)
        print("警告：未找到中文字体，中文标签可能显示异常")
        return ImageFont.load_default()

    def _draw_annotations(self, img, boxes, class_ids, confidences):
        img_copy = img.copy()
        colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0), (0, 255, 255)]
        
        for i in range(len(boxes)):
            # 强制提取边界框数值（解决嵌套列表问题）
            box = boxes[i].flatten() if isinstance(boxes[i], np.ndarray) else boxes[i]
            x1, y1, x2, y2 = map(int, [box[0], box[1], box[2], box[3]])  # 显式提取四个坐标
            class_id = class_ids[i]
            conf = confidences[i]
            class_name = self.class_names[int(class_id)]
            color = colors[i % len(colors)]
            
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
            
            label = f"{class_name} ({conf:.2f})"
            label_y = max(30, y1)
            if self.draw_chinese:
                img_pil = Image.fromarray(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(img_pil)
                draw.text((x1, label_y - 30), label, font=self.font, fill=(color[2], color[1], color[0]))
                img_copy = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            else:
                cv2.putText(img_copy, label, (x1, label_y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        return img_copy

    def _box_to_polygon(self, box):
        """将边界框转换为LabelMe格式的矩形多边形坐标"""
        # 关键修复：强制展平数组并提取数值，解决嵌套列表问题
        if isinstance(box, np.ndarray):
            box = box.flatten().tolist()  # 展平数组并转为列表
        else:
            box = box[:4]  # 确保只取前四个元素
        
        # 强制提取四个坐标值（防止任何嵌套结构）
        x1 = float(box[0]) if not isinstance(box[0], list) else float(box[0][0])
        y1 = float(box[1]) if not isinstance(box[1], list) else float(box[1][0])
        x2 = float(box[2]) if not isinstance(box[2], list) else float(box[2][0])
        y2 = float(box[3]) if not isinstance(box[3], list) else float(box[3][0])
        
        return [
            [x1, y1],  # 左上角
            [x2, y1],  # 右上角
            [x2, y2],  # 右下角
            [x1, y2]   # 左下角
        ]

    def _save_labelme_json(self, save_path, boxes, class_names, img_path, img_shape):
        img_filename = os.path.basename(img_path)
        
        labelme_data = {
            "version": "5.4.1",
            "flags": {},
            "shapes": [],
            "imagePath": img_filename,
            "imageData": None,
            "imageHeight": img_shape[0],
            "imageWidth": img_shape[1],
            "image_path_list": [img_filename],
            "channels": 3
        }
        
        for box, class_name in zip(boxes, class_names):
            polygon = self._box_to_polygon(box)
            shape = {
                "label": class_name,
                "points": polygon,
                "group_id": None,
                "description": "",
                "shape_type": "polygon",
                "flags": {},
                "mask": None
            }
            labelme_data["shapes"].append(shape)
        
        with open(f"{save_path}.json", "w", encoding="utf-8") as f:
            json.dump(labelme_data, f, ensure_ascii=False, indent=2)

    def predict_single_image(self, img_path, save_dir=None):
        if not os.path.exists(img_path):
            print(f"错误：图像不存在 - {img_path}")
            return None

        results = self.model(img_path)
        result = results[0]
        
        # 解析结果（CPU转换）
        boxes = result.boxes.xyxy.cpu().numpy()  # 形状为 (N, 4)
        class_ids = result.boxes.cls.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        img_shape = result.orig_shape

        object_count = len(boxes)
        print(f"[{os.path.basename(img_path)}] 检测到 {object_count} 个目标：")
        for i in range(object_count):
            class_name = self.class_names[int(class_ids[i])]
            box = boxes[i]
            print(f"  目标 {i+1}：类别={class_name}，置信度={confidences[i]:.4f}，"
                  f"位置=({int(box[0])},{int(box[1])})-({int(box[2])},{int(box[3])})")

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            img_save_path = os.path.join(save_dir, f"{base_name}.jpg")
            json_save_path = os.path.join(save_dir, base_name)
            
            img = cv2.imread(img_path)
            if object_count > 0 and self.draw_chinese:
                img = self._draw_annotations(img, boxes, class_ids, confidences)
            
            # 转换边界框为多边形
            polygons = [self._box_to_polygon(box) for box in boxes]
            class_names = [self.class_names[int(cid)] for cid in class_ids]
            
            cv2.imwrite(img_save_path, img)
            self._save_labelme_json(json_save_path, polygons, class_names, img_path, img_shape)

        return {
            "image_path": img_path,
            "object_count": object_count,
            "objects": [{
                "box": boxes[i].tolist(),
                "class_id": int(class_ids[i]),
                "class_name": self.class_names[int(class_ids[i])],
                "confidence": float(confidences[i])
            } for i in range(object_count)]
        }

    def predict_single_folder(self, folder_path, save_root=None, recursive=False):
        if not os.path.isdir(folder_path):
            print(f"错误：文件夹不存在 - {folder_path}")
            return []

        img_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
        img_paths = []
        for ext in img_extensions:
            if recursive:
                img_paths.extend(glob(os.path.join(folder_path, '**', ext), recursive=True))
            else:
                img_paths.extend(glob(os.path.join(folder_path, ext)))

        if not img_paths:
            print(f"警告：文件夹中未找到图像 - {folder_path}")
            return []

        print(f"\n{'递归' if recursive else '非递归'}处理文件夹：{folder_path}，共{len(img_paths)}张图像")
        all_results = []
        for img_path in img_paths:
            save_dir = save_root
            if save_root:
                rel_path = os.path.relpath(os.path.dirname(img_path), folder_path)
                save_dir = os.path.join(save_root, rel_path)
            
            result = self.predict_single_image(img_path, save_dir)
            if result:
                all_results.append(result)

        return all_results

def parse_args():
    parser = argparse.ArgumentParser(description='YOLO目标检测预测脚本（输出LabelMe格式JSON）')
    parser.add_argument('--model', type=str, 
                        default=r"D:\Min\Projects\VSCodeProjects\ultralytics-main\weights\yolov8n.pt", 
                        help='训练好的检测模型路径')
    parser.add_argument('--img', type=str, default=r'D:\Min\Projects\VSCodeProjects\ultralytics-main\data\demo.jpg', 
                        help='单张图像路径（可选）')
    parser.add_argument('--folder', type=str, default=None, 
                        help='单个文件夹路径（可选）')
    parser.add_argument('--recursive', action='store_false', 
                        help='是否递归读取文件夹')
    parser.add_argument('--save', type=str, default=r'D:\Min\Projects\VSCodeProjects\ultralytics-main\results\test_imgs', 
                        help='结果保存根目录')
    parser.add_argument('--draw-chinese', action='store_false', 
                        help='是否绘制中文标签')
    return parser.parse_args()

def main():
    args = parse_args()

    if not any([args.img, args.folder]):
        print("错误：请指定预测对象（--img 或 --folder）")
        return

    save_root = None
    if args.save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_root = os.path.join(args.save, f"detect_predict_{timestamp}")
        print(f"结果将保存至：{save_root}（图像+LabelMe格式JSON）")

    predictor = YOLODetectorPredictor(args.model, draw_chinese=args.draw_chinese)

    if args.img:
        predictor.predict_single_image(args.img, save_root)

    if args.folder:
        predictor.predict_single_folder(args.folder, save_root, recursive=args.recursive)

if __name__ == '__main__':
    main()