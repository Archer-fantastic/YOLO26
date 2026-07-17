import sys
import os
import argparse

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO 姿态估计训练脚本")

    parser.add_argument("--data", type=str, required=True, help="数据集配置文件（data.yaml）路径")

    parser.add_argument("--model", type=str, default=None, help="模型配置（.yaml 或 .pt），不传则使用 --weights")
    parser.add_argument("--weights", type=str, required=True, help="预训练权重文件（.pt）")
    parser.add_argument("--resume", action="store_true", help="是否从上一次中断处继续训练")

    parser.add_argument("--epochs", type=int, default=100, help="训练轮次")
    parser.add_argument("--imgsz", type=int, default=640, help="输入图片尺寸（需为32的倍数）")
    parser.add_argument("--batch", type=int, default=16, help="批次大小")
    parser.add_argument("--workers", type=int, default=0, help="数据加载线程数（Windows建议0）")
    parser.add_argument("--device", type=str, default="0", help="训练设备（cpu 或 0,1... 表示GPU）")

    parser.add_argument("--optimizer", type=str, default="auto", help="优化器: SGD/Adam/AdamW/RMSProp/auto")
    parser.add_argument("--lr0", type=float, default=0.01, help="初始学习率")
    parser.add_argument("--lrf", type=float, default=0.01, help="最终学习率比例（lr0*lrf）")
    parser.add_argument("--momentum", type=float, default=0.937, help="SGD动量或Adam beta1")
    parser.add_argument("--weight-decay", dest="weight_decay", type=float, default=0.0005, help="权重衰减")
    parser.add_argument("--warmup-epochs", dest="warmup_epochs", type=float, default=3.0, help="warmup epochs")
    parser.add_argument("--patience", type=int, default=100, help="早停耐心（epochs）")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--cos-lr", dest="cos_lr", action="store_true", help="启用余弦学习率")
    parser.set_defaults(cos_lr=False)
    parser.add_argument("--close-mosaic", dest="close_mosaic", type=int, default=10, help="最后N轮关闭mosaic")
    parser.add_argument("--no-close-mosaic", dest="close_mosaic", action="store_const", const=0, help="训练全程开启mosaic")

    parser.add_argument("--pretrained", dest="pretrained", action="store_true", help="启用pretrained")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false", help="关闭pretrained")
    parser.set_defaults(pretrained=True)

    parser.add_argument("--cache", dest="cache", action="store_true", help="启用cache")
    parser.add_argument("--no-cache", dest="cache", action="store_false", help="关闭cache")
    parser.set_defaults(cache=False)

    parser.add_argument("--amp", dest="amp", action="store_true", help="启用AMP")
    parser.add_argument("--no-amp", dest="amp", action="store_false", help="关闭AMP")
    parser.set_defaults(amp=False)

    parser.add_argument("--save", dest="save", action="store_true", help="保存训练结果")
    parser.add_argument("--no-save", dest="save", action="store_false", help="不保存训练结果")
    parser.set_defaults(save=True)

    parser.add_argument("--val", dest="val", action="store_true", help="训练中进行验证")
    parser.add_argument("--no-val", dest="val", action="store_false", help="训练中不进行验证")
    parser.set_defaults(val=True)

    parser.add_argument("--pose", type=float, default=12.0, help="pose loss gain")
    parser.add_argument("--kobj", type=float, default=1.0, help="keypoint objectness loss gain")

    parser.add_argument("--hsv-h", dest="hsv_h", type=float, default=0.015, help="HSV hue增强")
    parser.add_argument("--hsv-s", dest="hsv_s", type=float, default=0.7, help="HSV saturation增强")
    parser.add_argument("--hsv-v", dest="hsv_v", type=float, default=0.4, help="HSV value增强")
    parser.add_argument("--degrees", type=float, default=0.0, help="旋转角度")
    parser.add_argument("--translate", type=float, default=0.1, help="平移比例")
    parser.add_argument("--scale", type=float, default=0.5, help="缩放比例")
    parser.add_argument("--shear", type=float, default=0.0, help="错切角度")
    parser.add_argument("--perspective", type=float, default=0.0, help="透视变换比例")
    parser.add_argument("--flipud", type=float, default=0.0, help="上下翻转概率")
    parser.add_argument("--fliplr", type=float, default=0.5, help="左右翻转概率")
    parser.add_argument("--mosaic", type=float, default=1.0, help="mosaic概率")
    parser.add_argument("--mixup", type=float, default=0.0, help="mixup概率")

    parser.add_argument("--project", type=str, required=True, help="训练结果保存根目录")
    parser.add_argument("--name", type=str, default="train", help="本次训练文件夹名称")

    args = parser.parse_args()
    if args.model is None:
        args.model = args.weights
    return args


def train_pose(args):
    if args.resume:
        checkpoint_path = None
        if args.weights and os.path.exists(args.weights):
            checkpoint_path = args.weights
        if not checkpoint_path:
            potential_last = os.path.join(args.project, args.name, "weights", "last.pt")
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

    model = YOLO(args.model)
    if args.weights and args.model != args.weights and os.path.exists(args.weights):
        model.load(args.weights)

    if not os.path.exists(args.data):
        raise FileNotFoundError(f"数据集配置文件不存在: {args.data}")

    train_kwargs = {
        "task": "pose",
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "project": args.project,
        "name": args.name,
        "save": args.save,
        "val": args.val,
        "cache": args.cache,
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
        "pose": args.pose,
        "kobj": args.kobj,
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

    model.train(**train_kwargs)


if __name__ == "__main__":
    args = parse_args()
    train_pose(args)
