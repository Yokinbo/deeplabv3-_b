import os

# ----------------------------- 关键：WSL/服务器禁用GUI，避免Qt/xcb崩溃 -----------------------------
import matplotlib
matplotlib.use("Agg")  # 必须在导入 pyplot / show_results 之前
# ------------------------------------------------------------------------------------------------

from PIL import Image
from tqdm import tqdm

from deeplab import DeeplabV3
from utils.utils_metrics import compute_mIoU, show_results

'''
进行指标评估需要注意以下几点：
1、该文件生成的图为灰度图，因为值比较小，按照PNG形式的图看是没有显示效果的，所以看到近似全黑的图是正常的。
2、该文件计算的是验证集的miou，当前该库将测试集当作验证集使用，不单独划分测试集
'''
if __name__ == "__main__":
    #---------------------------------------------------------------------------#
    #   miou_mode用于指定该文件运行时计算的内容
    #   miou_mode为0代表整个miou计算流程，包括获得预测结果、计算miou。
    #   miou_mode为1代表仅仅获得预测结果。
    #   miou_mode为2代表仅仅计算miou。
    #---------------------------------------------------------------------------#
    miou_mode       = 0
    #------------------------------#
    #   分类个数+1、如2+1
    #------------------------------#
    num_classes     = 3
    #--------------------------------------------#
    #   区分的种类，和json_to_dataset里面的一样
    #--------------------------------------------#
    name_classes    = ["_background_","Openmine","PV"]
    #-------------------------------------------------------#
    #   指向VOC数据集所在的文件夹
    #-------------------------------------------------------#
    VOCdevkit_path  = 'VOCdevkit'

    image_ids       = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"),'r').read().splitlines()
    gt_dir          = os.path.join(VOCdevkit_path, "VOC2007/SegmentationClass/")
    miou_out_path   = "miou_out"
    pred_dir        = os.path.join(miou_out_path, 'detection-results')

    # 确保输出目录存在（即使只算miou也有地方落盘）
    os.makedirs(miou_out_path, exist_ok=True)

    if miou_mode == 0 or miou_mode == 1:
        os.makedirs(pred_dir, exist_ok=True)

        print("Load model.")
        deeplab = DeeplabV3()
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path  = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages", image_id + ".jpg")
            image       = Image.open(image_path)
            image       = deeplab.get_miou_png(image)
            image.save(os.path.join(pred_dir, image_id + ".png"))
        print("Get predict result done.")

    if miou_mode == 0 or miou_mode == 2:
        print("Get miou.")
        hist, IoUs, PA_Recall, Precision = compute_mIoU(
            gt_dir, pred_dir, image_ids, num_classes, name_classes
        )
        print("Get miou done.")

        # 额外保险：就算 show_results 内部异常，也至少写一个纯文本 summary
        try:
            show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)
            print("Save results done.")
        except Exception as e:
            print("[WARN] show_results failed, fallback to txt only:", repr(e))

        # 不依赖任何GUI的txt汇总（确保你一定能在miou_out里看到结果）
        try:
            import numpy as np
            IoUs_np = np.array(IoUs)
            PA_np = np.array(PA_Recall)
            Precision_np = np.array(Precision)

            summary_path = os.path.join(miou_out_path, "miou_summary.txt")
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(f"num_classes: {num_classes}\n")
                for i, cls in enumerate(name_classes):
                    f.write(
                        f"{cls:15s} "
                        f"IoU={IoUs_np[i]*100:.2f}  "
                        f"PA(Recall)={PA_np[i]*100:.2f}  "
                        f"Precision={Precision_np[i]*100:.2f}\n"
                    )
                f.write("\n")
                f.write(f"mIoU={IoUs_np.mean()*100:.2f}\n")
                f.write(f"mPA ={PA_np.mean()*100:.2f}\n")
                acc = (np.diag(hist).sum() / (hist.sum() + 1e-10)) * 100
                f.write(f"Accuracy={acc:.2f}\n")
            print(f"Write summary to {summary_path}")
        except Exception as e:
            print("[WARN] write summary failed:", repr(e))
