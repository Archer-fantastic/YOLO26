import sys
import os

# Ensure local ultralytics is used
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ultralytics import YOLO
import argparse

def parse_args():
    """解析命令行参数，支持灵活配置目标检测训练参数（含cache缓存配置）"""
    parser = argparse.ArgumentParser(description='YOLO 目标检测训练脚本（带缓存配置）')
    
    # 模型配置
    parser.add_argument('--model', type=str, default=None,
                        help='模型配置（.yaml 或 .pt），不传则使用 --weights')
    parser.add_argument('--weights', type=str, required=True,
                        help='预训练权重文件（.pt）')
    parser.add_argument('--resume', action='store_true', help='是否从上一次中断处继续训练')
    
    # 数据集配置
    parser.add_argument('--data', type=str, required=True,
                        help='数据集配置文件（data.yaml）路径')
    
    # 训练核心参数
    parser.add_argument('--epochs', type=int, default=100, 
                        help='训练轮次（快速验证设为2，正常训练30-300）')
    parser.add_argument('--imgsz', type=int, default=640, 
                        help='输入图片尺寸（需为32的倍数，检测常用640）')
    parser.add_argument('--batch', type=int, default=16, 
                        help='批次大小（显存不足减小，验证时设小一点）')
    parser.add_argument('--device', type=str, default='0', 
                        help='训练设备（"0"为GPU，"cpu"为CPU，多GPU用"0,1"）')
    parser.add_argument('--optimizer', type=str, default='auto', help='优化器: SGD/Adam/AdamW/RMSProp/auto')
    parser.add_argument('--lr0', type=float, default=0.01, help='初始学习率（默认0.01，小数据集可降至0.001）')
    parser.add_argument('--lrf', type=float, default=0.01, help='最终学习率比例（lr0*lrf）')
    parser.add_argument('--momentum', type=float, default=0.937, help='SGD动量或Adam beta1')
    parser.add_argument('--weight-decay', dest='weight_decay', type=float, default=0.0005, help='权重衰减')
    parser.add_argument('--warmup-epochs', dest='warmup_epochs', type=float, default=3.0, help='warmup epochs')
    parser.add_argument('--patience', type=int, default=100, help='早停耐心（epochs）')
    parser.add_argument('--seed', type=int, default=0, help='随机种子')
    parser.add_argument('--cos-lr', dest='cos_lr', action='store_true', help='启用余弦学习率')
    parser.set_defaults(cos_lr=False)
    parser.add_argument('--close-mosaic', dest='close_mosaic', type=int, default=10, help='最后N轮关闭mosaic')
    parser.add_argument('--no-close-mosaic', dest='close_mosaic', action='store_const', const=0, help='训练全程开启mosaic')
    
    # 缓存参数（核心新增）
    parser.add_argument('--cache', type=str, default='ram',
                        help='数据集缓存方式：True/ram=内存缓存，disk=磁盘缓存，False=不缓存')
    parser.add_argument('--no-cache', dest='cache', action='store_const', const='false',
                        help='关闭缓存（等价于 --cache false）')
    
    # 其他数据处理参数
    parser.add_argument('--workers', type=int, default=0, 
                        help='数据加载线程数（Windows建议0，Linux可设4-8）')

    parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='启用pretrained')
    parser.add_argument('--no-pretrained', dest='pretrained', action='store_false', help='关闭pretrained')
    parser.set_defaults(pretrained=True)

    parser.add_argument('--amp', dest='amp', action='store_true', help='启用AMP')
    parser.add_argument('--no-amp', dest='amp', action='store_false', help='关闭AMP')
    parser.set_defaults(amp=False)

    parser.add_argument('--save', dest='save', action='store_true', help='保存训练结果')
    parser.add_argument('--no-save', dest='save', action='store_false', help='不保存训练结果')
    parser.set_defaults(save=True)

    parser.add_argument('--val', dest='val', action='store_true', help='训练中进行验证')
    parser.add_argument('--no-val', dest='val', action='store_false', help='训练中不进行验证')
    parser.set_defaults(val=True)

    parser.add_argument('--hsv-h', dest='hsv_h', type=float, default=0.015, help='HSV hue增强')
    parser.add_argument('--hsv-s', dest='hsv_s', type=float, default=0.7, help='HSV saturation增强')
    parser.add_argument('--hsv-v', dest='hsv_v', type=float, default=0.4, help='HSV value增强')
    parser.add_argument('--degrees', type=float, default=0.0, help='旋转角度')
    parser.add_argument('--translate', type=float, default=0.1, help='平移比例')
    parser.add_argument('--scale', type=float, default=0.5, help='缩放比例')
    parser.add_argument('--shear', type=float, default=0.0, help='错切角度')
    parser.add_argument('--perspective', type=float, default=0.0, help='透视变换比例')
    parser.add_argument('--flipud', type=float, default=0.0, help='上下翻转概率')
    parser.add_argument('--fliplr', type=float, default=0.5, help='左右翻转概率')
    parser.add_argument('--mosaic', type=float, default=1.0, help='mosaic概率')
    parser.add_argument('--mixup', type=float, default=0.0, help='mixup概率')
    
    # 输出配置
    parser.add_argument('--project', type=str, required=True, help='训练结果保存根目录')
    parser.add_argument('--name', type=str, default='train', help='本次训练文件夹名称（用于区分实验）')
    
    args = parser.parse_args()
    if args.model is None:
        args.model = args.weights
    return args

def train_detection(args):
    """训练YOLO目标检测模型（支持缓存配置）"""
    if args.resume:
        checkpoint_path = None
        if args.weights and os.path.exists(args.weights):
            checkpoint_path = args.weights
        if not checkpoint_path:
            potential_last = os.path.join(args.project, args.name, 'weights', 'last.pt')
            if os.path.exists(potential_last):
                checkpoint_path = potential_last
        if checkpoint_path:
            print(f"🔄 正在从断点恢复训练: {checkpoint_path}")
            try:
                model = YOLO(checkpoint_path)
                model.train(resume=True)
                return
            except Exception as e:
                print(f"❌ 恢复训练失败: {e}")
                print("⚠️  将尝试重新开始训练...")
        else:
            print("⚠️  未找到可恢复的断点 (last.pt)，将开始新的训练...")

    # 1. 加载预训练检测模型
    model = YOLO(args.model)
    if args.weights and args.model != args.weights and os.path.exists(args.weights):
        model.load(args.weights)
    if args.device != 'cpu':
        model.to(f'cuda:{args.device}' if args.device else 'cuda')
    print(f"已加载模型: {args.model}，训练设备: {args.device}")

    # 2. 检查数据集配置文件
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"数据集配置文件不存在: {args.data}")
    print(f"使用数据集配置: {args.data}")

    # 3. 解析缓存参数（转换为YOLO支持的格式）
    # 支持：True/'ram'（内存）、'disk'（磁盘）、False（不缓存）
    cache_mode = args.cache
    if cache_mode.lower() == 'true':
        cache_mode = True
    elif cache_mode.lower() == 'false':
        cache_mode = False
    print(f"数据集缓存方式: {cache_mode}（内存缓存适合小数据集，磁盘缓存适合大数据集）")

    # 4. 开始训练
    print("开始训练目标检测模型...")
    train_kwargs = {
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "cache": cache_mode,
        "project": args.project,
        "name": args.name,
        "save": args.save,
        "val": args.val,
        "amp": args.amp,
        "pretrained": args.pretrained,
        "patience": args.patience,
        "seed": args.seed,
        "cos_lr": args.cos_lr,
        "close_mosaic": args.close_mosaic,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "hsv_h": args.hsv_h,
        "hsv_s": args.hsv_s,
        "hsv_v": args.hsv_v,
        "degrees": args.degrees,
        "translate": args.translate,
        "scale": args.scale,
        "shear": args.shear,
        "perspective": args.perspective,
        "flipud": args.flipud,
        "fliplr": args.fliplr,
        "mosaic": args.mosaic,
        "mixup": args.mixup,
    }
    results = model.train(
        **train_kwargs
    )

    # 5. 验证集评估
    print("训练完成，开始在验证集上评估...")
    metrics = model.val()  # 输出mAP等指标
    print(f"验证集指标: {metrics}")

    # 6. 示例图片推理
    with open(args.data, 'r', encoding='utf-8') as f:
        import yaml
        data_cfg = yaml.safe_load(f)
        val_img_dir = os.path.join(os.path.dirname(args.data), data_cfg.get('val', ''))

    if os.path.exists(val_img_dir) and os.listdir(val_img_dir):
        sample_img = os.path.join(val_img_dir, os.listdir(val_img_dir)[0])
        print(f"对示例图片推理: {sample_img}")
        pred = model(sample_img)
        save_dir = os.path.join(args.project, args.name, 'predictions')
        os.makedirs(save_dir, exist_ok=True)
        pred[0].save(os.path.join(save_dir, 'sample_pred.jpg'))
        print(f"推理结果已保存至: {save_dir}")

if __name__ == '__main__':
    args = parse_args()
    train_detection(args)
