import os

# WSL/服务器环境通常没有 GUI，这里提前切到 Agg，避免 matplotlib Qt/xcb 报错。
import matplotlib

matplotlib.use("Agg")

import numpy as np
from tqdm import tqdm

from deeplab import DeeplabV3
from multispectral_config import image_ext, in_channels, selected_bands, trained_model_path
from utils.utils_metrics import compute_mIoU, show_results

"""
DeepLabV3+ 多光谱 mIoU 评估脚本。

这个脚本适合当前 Sentinel-2 光伏提取任务：
1. 数据仍然使用 VOCdevkit/VOC2007 目录结构。
2. 影像可以是 .tif 多波段 patch。
3. 波段模式、输入通道数、权重路径统一从 multispectral_config.py 读取。
4. 预测阶段直接把 tif 路径交给 DeeplabV3，由 deeplab.py 内部用 rasterio 读取。

常用流程：
- miou_mode = 0：先生成预测 mask，再计算 mIoU
- miou_mode = 1：只生成预测 mask
- miou_mode = 2：只计算 mIoU，适合预测结果已经生成好的情况
"""


def write_summary(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes):
    # show_results 会生成图表；这里额外写一个纯文本结果，方便没有 GUI 的环境查看。
    IoUs_np = np.array(IoUs)
    PA_np = np.array(PA_Recall)
    Precision_np = np.array(Precision)

    summary_path = os.path.join(miou_out_path, "miou_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"num_classes: {len(name_classes)}\n")
        f.write(f"model_path: {trained_model_path}\n")
        f.write(f"image_ext: {image_ext}\n")
        f.write(f"selected_bands: {selected_bands}\n")
        f.write(f"in_channels: {in_channels}\n\n")

        for i, cls in enumerate(name_classes):
            f.write(
                f"{cls:15s} "
                f"IoU={IoUs_np[i] * 100:.2f}  "
                f"PA(Recall)={PA_np[i] * 100:.2f}  "
                f"Precision={Precision_np[i] * 100:.2f}\n"
            )

        f.write("\n")
        f.write(f"mIoU={IoUs_np.mean() * 100:.2f}\n")
        f.write(f"mPA ={PA_np.mean() * 100:.2f}\n")
        acc = (np.diag(hist).sum() / (hist.sum() + 1e-10)) * 100
        f.write(f"Accuracy={acc:.2f}\n")

    print(f"Write summary to {summary_path}")


if __name__ == "__main__":
    # ---------------------------------------------------------------------------#
    #   miou_mode 用于指定该文件运行时计算的内容：
    #   0 = 生成预测结果 + 计算 mIoU
    #   1 = 只生成预测结果
    #   2 = 只计算 mIoU
    # ---------------------------------------------------------------------------#
    miou_mode = 0

    # ------------------------------#
    #   分类个数，包含背景类
    # ------------------------------#
    num_classes = 2

    # --------------------------------------------#
    #   类别名称，顺序要和标签像素值一致
    # --------------------------------------------#
    name_classes = ["_background_", "PV"]

    # -------------------------------------------------------#
    #   VOC 数据集根目录
    # -------------------------------------------------------#
    VOCdevkit_path = "VOCdevkit"

    # -------------------------------------------------------#
    #   评估哪个划分文件
    #   常用：val.txt / test.txt
    # -------------------------------------------------------#
    image_set = "test.txt"

    image_ids = open(
        os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation", image_set),
        "r",
        encoding="utf-8",
    ).read().splitlines()

    gt_dir = os.path.join(VOCdevkit_path, "VOC2007/SegmentationClass")
    miou_out_path = "miou_out"
    pred_dir = os.path.join(miou_out_path, "detection-results")

    os.makedirs(miou_out_path, exist_ok=True)

    if miou_mode == 0 or miou_mode == 1:
        os.makedirs(pred_dir, exist_ok=True)

        print("Load model.")
        deeplab = DeeplabV3(
            model_path=trained_model_path,
            num_classes=num_classes,
            image_ext=image_ext,
            selected_bands=selected_bands,
            in_channels=in_channels,
        )
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path = os.path.join(
                VOCdevkit_path,
                "VOC2007/JPEGImages",
                image_id + image_ext,
            )
            image = deeplab.get_miou_png(image_path)
            image.save(os.path.join(pred_dir, image_id + ".png"))
        print("Get predict result done.")

    if miou_mode == 0 or miou_mode == 2:
        print("Get miou.")
        hist, IoUs, PA_Recall, Precision = compute_mIoU(
            gt_dir, pred_dir, image_ids, num_classes, name_classes
        )
        print("Get miou done.")

        try:
            show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)
            print("Save result figures done.")
        except Exception as e:
            print("[WARN] show_results failed, fallback to txt only:", repr(e))

        try:
            write_summary(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)
        except Exception as e:
            print("[WARN] write summary failed:", repr(e))
