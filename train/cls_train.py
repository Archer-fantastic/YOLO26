import argparse
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ultralytics import YOLO


def parse_args():
    """解析命令行参数，支持灵活配置训练参数."""
    parser = argparse.ArgumentParser(description="YOLO 图像分类训练脚本")

    # 模型配置
    parser.add_argument("--model", type=str, default=None, help="模型配置（.yaml 或 .pt），不传则使用 --weights")
    parser.add_argument("--weights", type=str, required=True, help="预训练权重文件（.pt）")
    parser.add_argument("--resume", action="store_true", help="是否从上一次中断处继续训练")

    # 数据集配置
    parser.add_argument(
        "--data", type=str, required=True, help="数据集根目录（需包含train/val文件夹，每个类别一个子文件夹）"
    )

    # 训练核心参数
    parser.add_argument("--epochs", type=int, default=300, help="训练轮次（小数据集30-100，大数据集可增加）")
    parser.add_argument("--imgsz", type=int, default=320, help="输入图片尺寸（需为32的倍数，如224/320/640）")
    parser.add_argument("--batch", type=int, default=16, help="批次大小（GPU显存不足时减小，如8/4）")
    parser.add_argument("--device", type=str, default="0", help='训练设备（"0"为GPU，"cpu"为CPU）')
    parser.add_argument("--optimizer", type=str, default="auto", help="优化器: SGD/Adam/AdamW/RMSProp/auto")
    parser.add_argument("--lr0", type=float, default=0.01, help="初始学习率（默认0.01，小数据集可减小至0.001）")
    parser.add_argument("--lrf", type=float, default=0.01, help="最终学习率比例（lr0*lrf）")
    parser.add_argument("--momentum", type=float, default=0.937, help="SGD动量或Adam beta1")
    parser.add_argument("--weight-decay", dest="weight_decay", type=float, default=0.0005, help="权重衰减")
    parser.add_argument("--warmup-epochs", dest="warmup_epochs", type=float, default=3.0, help="warmup epochs")
    parser.add_argument("--patience", type=int, default=100, help="早停耐心（epochs）")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--cos-lr", dest="cos_lr", action="store_true", help="启用余弦学习率")
    parser.set_defaults(cos_lr=False)

    parser.add_argument("--cache", dest="cache", action="store_true", help="启用cache")
    parser.add_argument("--no-cache", dest="cache", action="store_false", help="关闭cache")
    parser.set_defaults(cache=True)

    parser.add_argument("--amp", dest="amp", action="store_true", help="启用AMP")
    parser.add_argument("--no-amp", dest="amp", action="store_false", help="关闭AMP")
    parser.set_defaults(amp=False)

    parser.add_argument("--save", dest="save", action="store_true", help="保存训练结果")
    parser.add_argument("--no-save", dest="save", action="store_false", help="不保存训练结果")
    parser.set_defaults(save=True)

    parser.add_argument("--val", dest="val", action="store_true", help="训练中进行验证")
    parser.add_argument("--no-val", dest="val", action="store_false", help="训练中不进行验证")
    parser.set_defaults(val=True)

    parser.add_argument("--pretrained", dest="pretrained", action="store_true", help="启用pretrained")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false", help="关闭pretrained")
    parser.set_defaults(pretrained=True)

    parser.add_argument("--dropout", type=float, default=0.0, help="分类头 dropout")
    parser.add_argument(
        "--auto-augment",
        dest="auto_augment",
        type=str,
        default="randaugment",
        help="分类增强策略: randaugment/autoaugment/augmix",
    )
    parser.add_argument("--erasing", type=float, default=0.4, help="Random Erasing 概率")
    parser.add_argument("--mixup", type=float, default=0.0, help="mixup概率")
    parser.add_argument("--cutmix", type=float, default=0.0, help="cutmix概率")

    # 输出配置
    parser.add_argument("--project", type=str, required=True, help="训练结果保存根目录")
    parser.add_argument("--name", type=str, default="train", help="本次训练文件夹名称（用于区分不同实验）")

    args = parser.parse_args()
    if args.model is None:
        args.model = args.weights
    return args


def train_classification(args):
    """训练YOLO图像分类模型."""
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

    # 1. 加载预训练分类模型
    # - 模型后缀为-cls，n/s/m/l/x代表尺寸（nano到xlarge）
    model = YOLO(args.model)
    if args.weights and args.model != args.weights and os.path.exists(args.weights):
        model.load(args.weights)
    print(f"已加载模型: {args.model}")

    # 2. 检查数据集目录是否存在
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"数据集目录不存在: {args.data}")
    print(f"使用数据集: {args.data}")

    # 3. 开始训练
    print("开始训练图像分类模型...")
    train_kwargs = {
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "optimizer": args.optimizer,
        "project": args.project,
        "name": args.name,
        "cache": args.cache,
        "amp": args.amp,
        "save": args.save,
        "val": args.val,
        "pretrained": args.pretrained,
        "patience": args.patience,
        "seed": args.seed,
        "cos_lr": args.cos_lr,
        "dropout": args.dropout,
        "auto_augment": args.auto_augment,
        "erasing": args.erasing,
        "mixup": args.mixup,
        "cutmix": args.cutmix,
    }
    model.train(**train_kwargs)

    # 4. 训练完成后在验证集上评估
    print("训练完成，开始在验证集上评估...")
    metrics = model.val()  # 评估指标包括准确率（top1、top5等）
    print(f"验证集指标: {metrics}")

    # 5. 对示例图片进行推理（可选）
    val_img_dir = os.path.join(args.data, "val")  # 验证集图像根目录
    if os.path.exists(val_img_dir):
        # 找第一个类别文件夹中的第一张图片
        class_dirs = [d for d in os.listdir(val_img_dir) if os.path.isdir(os.path.join(val_img_dir, d))]
        if class_dirs:
            sample_img_path = os.path.join(val_img_dir, class_dirs[0])
            sample_imgs = [f for f in os.listdir(sample_img_path) if f.endswith((".jpg", ".png"))]
            if sample_imgs:
                sample_img = os.path.join(sample_img_path, sample_imgs[0])
                print(f"对示例图片推理: {sample_img}")
                pred = model(sample_img)
                # 保存推理结果
                save_dir = os.path.join(args.project, args.name, "predictions")
                os.makedirs(save_dir, exist_ok=True)
                pred[0].save(os.path.join(save_dir, "sample_pred.jpg"))
                print(f"推理结果已保存至: {save_dir}")


if __name__ == "__main__":
    args = parse_args()
    train_classification(args)
