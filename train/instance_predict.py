from ultralytics import YOLO
import os
import argparse
import cv2
import numpy as np
import json
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from glob import glob

class YOLOSegmentPredictor:
    def __init__(self, model_path, draw_chinese=False, save_vis=False, draw_box=False, draw_label=False):
        """初始化实例分割预测器"""
        self.model = YOLO(model_path)
        self.class_names = self.model.names
        self.draw_chinese = draw_chinese
        self.save_vis = save_vis
        self.draw_box = draw_box
        self.draw_label = draw_label
        print(f"已加载实例分割模型：{model_path}，包含类别：{list(self.class_names.values())}")
        print(f"中文标签绘制：{'开启' if draw_chinese else '关闭'}")
        print(f"可视化结果保存：{'开启' if save_vis else '关闭'}")
        print(f"绘制矩形框：{'开启' if draw_box else '关闭'}")
        print(f"绘制文字标签：{'开启' if draw_label else '关闭'}")
        self.font = self._load_chinese_font() if draw_chinese else None

    def _load_chinese_font(self):
        """加载支持中文的字体"""
        # 1. 尝试使用 matplotlib 查找系统中的中文字体 (最稳健的方法)
        try:
            from matplotlib.font_manager import fontManager
            # 查找包含特定关键字的中文字体
            candidates = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'FangSong', 'PingFang', 'Heiti']
            for f in fontManager.ttflist:
                for candidate in candidates:
                    if candidate.lower() in f.name.lower():
                        print(f"通过 matplotlib 找到字体：{f.name} ({f.fname})")
                        return ImageFont.truetype(f.fname, 30)
        except Exception as e:
            print(f"matplotlib 字体查找失败: {e}")

        # 2. 尝试常见的字体文件名（PIL会自动在系统字体目录搜索）
        common_fonts = ["simhei.ttf", "msyh.ttc", "simsun.ttc", "arialuni.ttf"]
        for font_name in common_fonts:
            try:
                font = ImageFont.truetype(font_name, 30)
                print(f"成功加载字体：{font_name}")
                return font
            except Exception:
                continue

        # 3. 尝试绝对路径（针对部分Windows环境）
        abs_paths = [
            r"C:\Windows\Fonts\simhei.ttf",
            r"C:\Windows\Fonts\msyh.ttc",
            r"C:\Windows\Fonts\simsun.ttc",
            r"/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            r"/System/Library/Fonts/PingFang.ttc"
        ]
        for path in abs_paths:
            if os.path.exists(path):
                try:
                    font = ImageFont.truetype(path, 30)
                    print(f"成功加载字体：{path}")
                    return font
                except Exception:
                    continue

        print("警告：未找到中文字体，中文标签将显示为 '????'")
        return ImageFont.load_default()

    def _draw_annotations(self, img, boxes, masks, class_ids, confidences, img_shape):
        """绘制分割结果"""
        img_copy = img.copy()
        colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0), (0, 255, 255)]
        orig_h, orig_w = img_shape
        
        for i in range(len(boxes)):
            x1, y1, x2, y2 = map(int, boxes[i])
            mask = masks[i]
            class_id = class_ids[i]
            conf = confidences[i]
            class_name = self.class_names[class_id]
            color = colors[i % len(colors)]
            
            # 调整掩码尺寸
            mask_resized = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            mask_3d = np.stack([mask_resized]*3, axis=-1)
            
            # 绘制掩码
            img_copy = np.where(mask_3d, 
                               cv2.addWeighted(img_copy, 0.7, np.full_like(img_copy, color), 0.3, 0),
                               img_copy)
            
            # 绘制边界框
            if self.draw_box:
                cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
            
            # 绘制标签（仅当 draw_label 为 True 时）
            if self.draw_label:
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

    def _mask_to_polygon(self, mask, img_shape):
        """
        将掩码转换为LabelMe格式的多边形坐标
        :param mask: 二值掩码（模型输出）
        :param img_shape: 原始图像尺寸 (height, width)
        :return: 多边形坐标列表 [[x1,y1], [x2,y2], ...]
        """
        orig_h, orig_w = img_shape
        # 调整掩码到原始图像尺寸
        mask_resized = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        
        # 查找轮廓
        contours, _ = cv2.findContours(
            mask_resized.astype(np.uint8) * 255,
            cv2.RETR_EXTERNAL,  # 只保留最外层轮廓
            cv2.CHAIN_APPROX_SIMPLE  # 简化轮廓
        )
        
        if not contours:
            return []
        
        # 取最大的轮廓（防止噪声）
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 转换为坐标列表
        polygon = largest_contour.squeeze().tolist()
        # 确保是二维列表 [[x1,y1], [x2,y2], ...]
        if isinstance(polygon[0], int):
            polygon = [polygon]
        return polygon

    def _save_labelme_json(self, save_path, polygons, class_names, img_path, img_shape):
        """
        保存LabelMe格式的JSON文件
        :param save_path: JSON保存路径（不含后缀）
        :param polygons: 多边形坐标列表（每个实例一个多边形）
        :param class_names: 类别名称列表
        :param img_path: 原始图像路径
        :param img_shape: 图像尺寸 (height, width)
        """
        # 获取图像文件名
        img_filename = os.path.basename(img_path)
        
        # 构建LabelMe格式数据
        labelme_data = {
            "version": "5.4.1",
            "flags": {},
            "shapes": [],
            "imagePath": img_filename,
            "imageData": None,  # 不保存图像二进制数据
            "imageHeight": img_shape[0],
            "imageWidth": img_shape[1],
            "image_path_list": [img_filename],
            "channels": 3  # 默认RGB通道
        }
        
        # 添加每个实例的多边形信息
        for i, (polygon, class_name) in enumerate(zip(polygons, class_names)):
            if not polygon:
                continue  # 跳过空多边形
            
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
        
        # 保存JSON
        with open(f"{save_path}.json", "w", encoding="utf-8") as f:
            json.dump(labelme_data, f, ensure_ascii=False, indent=2)

    def predict_single_image(self, img_path, save_dir=None, vis_dir=None):
        """单张图像实例分割预测"""
        if not os.path.exists(img_path):
            print(f"错误：图像不存在 - {img_path}")
            return None

        # 执行预测
        results = self.model(img_path)
        result = results[0]
        
        # 解析结果（CPU转换）
        boxes = result.boxes.xyxy.cpu().numpy()
        masks = result.masks.data.cpu().numpy() if result.masks is not None else []
        class_ids = result.boxes.cls.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        img_shape = result.orig_shape  # (height, width)

        # 终端输出
        instance_count = len(boxes)
        print(f"[{os.path.basename(img_path)}] 检测到 {instance_count} 个实例：")
        for i in range(instance_count):
            class_name = self.class_names[int(class_ids[i])]
            print(f"  实例 {i+1}：类别={class_name}，置信度={confidences[i]:.4f}")

        # 保存结果
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            img_save_path = os.path.join(save_dir, f"{base_name}.jpg")
            json_save_path = os.path.join(save_dir, base_name)
            
            # 读取原图
            img = cv2.imread(img_path)
            
            # 保存可视化结果（如果开启且提供了vis_dir）
            if self.save_vis and instance_count > 0 and vis_dir:
                os.makedirs(vis_dir, exist_ok=True)
                vis_path = os.path.join(vis_dir, f"{base_name}.jpg")
                vis_img = self._draw_annotations(img, boxes, masks, class_ids, confidences, img_shape)
                cv2.imwrite(vis_path, vis_img)
            
            # 转换掩码为多边形（LabelMe格式）
            polygons = []
            class_names = []
            for i in range(instance_count):
                mask = masks[i] if i < len(masks) else None
                if mask is not None:
                    polygon = self._mask_to_polygon(mask, img_shape)
                    polygons.append(polygon)
                    class_names.append(self.class_names[int(class_ids[i])])
            
            # 保存原图和LabelMe格式JSON
            cv2.imwrite(img_save_path, img)
            self._save_labelme_json(json_save_path, polygons, class_names, img_path, img_shape)

        return {
            "image_path": img_path,
            "instance_count": instance_count,
            "instances": [{
                "box": boxes[i].tolist(),
                "class_id": int(class_ids[i]),
                "class_name": self.class_names[int(class_ids[i])],
                "confidence": float(confidences[i])
            } for i in range(instance_count)]
        }

    def predict_single_folder(self, folder_path, save_root=None, recursive=False):
        """单个文件夹实例分割预测"""
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
            # 计算保存路径
            rel_path = os.path.relpath(os.path.dirname(img_path), folder_path)
            
            # predict 文件夹路径（原图+JSON）
            predict_dir = os.path.join(save_root, 'predict', rel_path)
            
            # vis 文件夹路径（可视化图）
            vis_dir = os.path.join(save_root, 'vis', rel_path)

            result = self.predict_single_image(img_path, predict_dir, vis_dir)
            if result:
                all_results.append(result)

        return all_results

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='YOLO实例分割预测脚本（输出LabelMe格式JSON）')
    parser.add_argument('--model', type=str, 
                        default=r"D:\Min\Projects\VSCodeProjects\dataset\instance_HD\instance_HD_yolo\yolov11n\train\weights\best.pt", 
                        help='训练好的分割模型路径')
    parser.add_argument('--img', type=str, default=None, 
                        help='单张图像路径（可选）')
    parser.add_argument('--folder', type=str, default=r'D:\Min\Projects\VSCodeProjects\dataset\instance_HD\测试集', 
                        help='单个文件夹路径（可选）')
    parser.add_argument('--recursive', action='store_false', 
                        help='是否递归读取文件夹')
    parser.add_argument('--save', type=str, default=r'D:\Min\Projects\VSCodeProjects\dataset\instance_HD\instance_HD_yolo\yolov11n\train\测试结果', 
                        help='结果保存根目录')
    # 修改默认值为开启，参数改为 --no-vis 用于关闭
    parser.add_argument('--no-vis', action='store_true', 
                        help='是否关闭保存可视化结果（默认开启）')
    # 修改默认值为关闭，参数改为 --draw-chinese 用于开启
    parser.add_argument('--draw-chinese', action='store_true', 
                        help='是否开启中文标签绘制（默认关闭）')
    parser.add_argument('--box', action='store_true', 
                        help='是否绘制矩形框（默认关闭）')
    parser.add_argument('--label', action='store_true', 
                        help='是否绘制文字标签（类别+置信度）（默认关闭）')
    return parser.parse_args()

def main():
    args = parse_args()

    if not any([args.img, args.folder]):
        print("错误：请指定预测对象（--img 或 --folder）")
        return

    save_root = None
    if args.save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_root = os.path.join(args.save, f"segment_predict_{timestamp}")
        print(f"结果将保存至：{save_root}（predict=原图+JSON, vis=可视化图）")

    # 配置预测器：
    # draw_chinese: 仅当需要绘制标签时生效，但参数仍保留
    # save_vis: 默认开启，除非指定 --no-vis
    # draw_box: 默认关闭，指定 --box 开启
    # draw_label: 默认关闭，指定 --label 开启
    predictor = YOLOSegmentPredictor(args.model, draw_chinese=args.draw_chinese, save_vis=not args.no_vis, draw_box=args.box, draw_label=args.label)

    if args.img:
        base_name = os.path.splitext(os.path.basename(args.img))[0]
        predict_dir = os.path.join(save_root, 'predict') if save_root else None
        vis_dir = os.path.join(save_root, 'vis') if save_root else None
        predictor.predict_single_image(args.img, predict_dir, vis_dir)

    if args.folder:
        predictor.predict_single_folder(args.folder, save_root, recursive=args.recursive)

if __name__ == '__main__':
    main()