import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from deeplab import DeeplabV3
from multispectral_config import image_ext, in_channels, selected_bands, trained_model_path


# ===================== 可编辑配置区 =====================
# 直接修改这里，然后运行：
#   python mymulti_predict.py
#
# input_path 支持：
#   1. 单张图片，例如 r"论文制图\测试图\原图\shenmu39.tif"
#   2. 图片文件夹，例如 r"论文制图\测试图\原图"
EDITABLE_CONFIG = {
    "input_path": r"论文制图\测试图\原图",
    "label_dir": r"论文制图\测试图\label标签",
    "weights": trained_model_path,
    "output_dir": r"论文制图\6band测试结果",
    "device": "cuda:0",
    "input_size": 256,
    "num_classes": 2,
    "pv_class": 1,
    "suffixes": [".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"],
    "label_suffixes": [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"],
    "save_mask": True,
    "save_overlay": True,
    "save_prob": False,
    "save_confusion": True,
}
# =======================================================


def time_synchronized():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def imwrite_unicode(path, image):
    """OpenCV on Windows may fail on Chinese paths; imencode+tofile is safer."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix or ".png"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise IOError(f"failed to encode image for: {path}")
    encoded.tofile(str(path))


def collect_input_images(input_path, suffixes):
    input_path = Path(input_path)
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        suffix_set = {suffix.lower() for suffix in suffixes}
        return sorted(
            p for p in input_path.iterdir()
            if p.is_file() and p.suffix.lower() in suffix_set
        )
    raise FileNotFoundError(f"input path does not exist: {input_path}")


def find_label_for_image(image_path, label_dir, label_suffixes):
    if not label_dir:
        return None

    label_dir = Path(label_dir)
    if not label_dir.exists():
        raise FileNotFoundError(f"label_dir does not exist: {label_dir}")

    for suffix in label_suffixes:
        label_path = label_dir / f"{image_path.stem}{suffix}"
        if label_path.exists():
            return label_path
    return None


def read_label_mask(label_path, target_shape):
    try:
        label = np.array(Image.open(label_path).convert("L"))
    except Exception as exc:
        raise FileNotFoundError(f"failed to read label: {label_path}") from exc

    if label.shape != target_shape:
        label = cv2.resize(
            label,
            (target_shape[1], target_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    return (label > 0).astype(np.uint8)


def save_overlay(preview_img, pred_mask, output_overlay):
    overlay = preview_img.copy()
    red = np.zeros_like(overlay)
    red[:, :, 0] = 255
    overlay = np.where(pred_mask[..., None] > 0, 0.55 * overlay + 0.45 * red, overlay)
    imwrite_unicode(output_overlay, cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_RGB2BGR))


def save_confusion_map(pred_mask, label_mask, output_confusion):
    """Save TP/FP/FN/TN color map. TN=black, TP=white, FP=blue, FN=red."""
    pred01 = (pred_mask > 0).astype(np.uint8)
    label01 = (label_mask > 0).astype(np.uint8)

    rgb = np.zeros((label01.shape[0], label01.shape[1], 3), dtype=np.uint8)
    tp = (pred01 == 1) & (label01 == 1)
    fp = (pred01 == 1) & (label01 == 0)
    fn = (pred01 == 0) & (label01 == 1)

    rgb[tp] = [255, 255, 255]
    rgb[fp] = [0, 0, 255]
    rgb[fn] = [255, 0, 0]
    imwrite_unicode(output_confusion, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def predict_mask_and_prob(deeplab, image_path, pv_class):
    image_data, nw, nh, original_h, original_w, preview_img = deeplab.prepare_input(str(image_path))

    with torch.no_grad():
        images = torch.from_numpy(image_data)
        if deeplab.cuda:
            images = images.cuda()

        t_start = time_synchronized()
        pred = deeplab.net(images)[0]
        t_end = time_synchronized()

        prob = F.softmax(pred.permute(1, 2, 0), dim=-1).cpu().numpy()

    top = int((deeplab.input_shape[0] - nh) // 2)
    left = int((deeplab.input_shape[1] - nw) // 2)
    prob = prob[top:top + nh, left:left + nw]
    prob = cv2.resize(prob, (original_w, original_h), interpolation=cv2.INTER_LINEAR)

    pred_mask = prob.argmax(axis=-1).astype(np.uint8)
    pv_prob = prob[:, :, pv_class] if pv_class < prob.shape[2] else None
    return pred_mask, pv_prob, np.array(preview_img), t_end - t_start


def predict_one_image(deeplab, image_path, args):
    pred_mask, pv_prob, preview_img, elapsed = predict_mask_and_prob(
        deeplab,
        image_path,
        args.pv_class,
    )

    output_dir = Path(args.output_dir)
    mask_path = output_dir / "模型预测mask_0-255" / f"{image_path.stem}_mask.png"
    overlay_path = output_dir / "红色半透明叠加图" / f"{image_path.stem}_overlay.png"
    prob_path = output_dir / "模型预测概率图" / f"{image_path.stem}_prob.png"
    label_view_path = output_dir / "人工标签可视化_0-255" / f"{image_path.stem}_label.png"
    confusion_path = output_dir / "TP_FP_FN_TN彩色误差图" / f"{image_path.stem}_confusion.png"

    binary_mask = (pred_mask == args.pv_class).astype(np.uint8)

    if args.save_mask:
        imwrite_unicode(mask_path, binary_mask * 255)

    if args.save_overlay:
        save_overlay(preview_img, binary_mask, overlay_path)

    if args.save_prob and pv_prob is not None:
        prob_img = np.clip(pv_prob * 255.0, 0, 255).astype(np.uint8)
        imwrite_unicode(prob_path, prob_img)

    if args.save_confusion:
        label_path = find_label_for_image(image_path, args.label_dir, args.label_suffixes)
        if label_path is None:
            print(f"[warn] no label found for {image_path.name}, skip confusion map")
        else:
            label_mask = read_label_mask(label_path, binary_mask.shape)
            imwrite_unicode(label_view_path, label_mask * 255)
            save_confusion_map(binary_mask, label_mask, confusion_path)

    print(f"[done] {image_path.name} inference={elapsed:.4f}s")


def main(args):
    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"weights file does not exist: {args.weights}")

    image_paths = collect_input_images(args.input_path, args.suffixes)
    if not image_paths:
        raise FileNotFoundError(f"No supported images found in: {args.input_path}")

    use_cuda = args.device.startswith("cuda") and torch.cuda.is_available()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA is unavailable, fallback to CPU")

    print("Current DeepLabV3+ multispectral prediction config:")
    print(f"  image_ext     : {image_ext}")
    print(f"  selected_bands: {selected_bands}")
    print(f"  in_channels   : {in_channels}")
    print(f"  weights       : {args.weights}")
    print(f"  input_path    : {args.input_path}")
    print(f"  label_dir     : {args.label_dir or '(disabled)'}")
    print(f"  image_count   : {len(image_paths)}")
    print(f"  output_dir    : {args.output_dir}")
    print(f"  device        : {'cuda' if use_cuda else 'cpu'}")

    deeplab = DeeplabV3(
        model_path=args.weights,
        num_classes=args.num_classes,
        image_ext=image_ext,
        selected_bands=selected_bands,
        in_channels=in_channels,
        input_shape=[args.input_size, args.input_size],
        cuda=use_cuda,
        mix_type=0,
    )

    for image_path in image_paths:
        predict_one_image(deeplab, image_path, args)

    print("Saved outputs to:", args.output_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="DeepLabV3+ multispectral batch prediction")
    parser.add_argument("--input-path", default=EDITABLE_CONFIG["input_path"], help="input image file or folder")
    parser.add_argument("--label-dir", default=EDITABLE_CONFIG["label_dir"], help="manual label folder")
    parser.add_argument("--weights", default=EDITABLE_CONFIG["weights"], help="model weights path")
    parser.add_argument("--output-dir", default=EDITABLE_CONFIG["output_dir"], help="output folder")
    parser.add_argument("--device", default=EDITABLE_CONFIG["device"], help="prediction device")
    parser.add_argument("--input-size", default=EDITABLE_CONFIG["input_size"], type=int, help="square inference input size")
    parser.add_argument("--num-classes", default=EDITABLE_CONFIG["num_classes"], type=int, help="number of classes")
    parser.add_argument("--pv-class", default=EDITABLE_CONFIG["pv_class"], type=int, help="foreground/PV class id")
    parser.add_argument("--suffixes", nargs="+", default=EDITABLE_CONFIG["suffixes"], help="image suffixes for folder input")
    parser.add_argument("--label-suffixes", nargs="+", default=EDITABLE_CONFIG["label_suffixes"], help="label suffixes")
    parser.add_argument("--save-mask", action="store_true", default=EDITABLE_CONFIG["save_mask"])
    parser.add_argument("--no-save-mask", action="store_false", dest="save_mask")
    parser.add_argument("--save-overlay", action="store_true", default=EDITABLE_CONFIG["save_overlay"])
    parser.add_argument("--no-save-overlay", action="store_false", dest="save_overlay")
    parser.add_argument("--save-prob", action="store_true", default=EDITABLE_CONFIG["save_prob"])
    parser.add_argument("--no-save-prob", action="store_false", dest="save_prob")
    parser.add_argument("--save-confusion", action="store_true", default=EDITABLE_CONFIG["save_confusion"])
    parser.add_argument("--no-save-confusion", action="store_false", dest="save_confusion")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
