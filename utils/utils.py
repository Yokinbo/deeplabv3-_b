import random

import numpy as np
import torch
from PIL import Image

try:
    # 多光谱任务会在项目根目录下提供 multispectral_config.py。
    # 训练、推理、验证都从同一个配置读取标准化参数，避免前后处理不一致。
    from multispectral_config import normalization_config
except Exception:
    # 保底配置：
    # 如果当前项目暂时没有多光谱配置文件，或者只跑旧 RGB 流程，
    # preprocess_input 仍然可以正常工作。
    normalization_config = {
        "reflectance_scale": 10000.0,
        "enable_clip": False,
        "clip_min": None,
        "clip_max": None,
        "enable_mean_std": False,
        "mean": None,
        "std": None,
    }

#---------------------------------------------------------#
#   将普通 PIL 图像转换成 RGB，防止灰度图在 RGB 流程里报错。
#   注意：多光谱 tif 不应该调用这个函数，否则会丢失 4/6 波段信息。
#---------------------------------------------------------#
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 

#---------------------------------------------------#
#   对输入图像进行resize
#---------------------------------------------------#
def resize_image(image, size):
    iw, ih  = image.size
    w, h    = size

    scale   = min(w/iw, h/ih)
    nw      = int(iw*scale)
    nh      = int(ih*scale)

    image   = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))

    return new_image, nw, nh
    
#---------------------------------------------------#
#   获得学习率
#---------------------------------------------------#
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

#---------------------------------------------------#
#   设置种子
#---------------------------------------------------#
def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#---------------------------------------------------#
#   设置Dataloader的种子
#---------------------------------------------------#
def worker_init_fn(worker_id, rank, seed):
    worker_seed = rank + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def _reshape_band_vector(values, channels, name):
    # 将 [C] 形式的波段参数整理成 [1, 1, C]。
    # 这样它可以直接和 HWC 格式影像广播运算：
    # image.shape = [H, W, C]
    # mean.shape  = [1, 1, C]
    if values is None:
        return None

    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 1 or arr.shape[0] != channels:
        raise ValueError(
            f"{name} 的长度应与输入通道数一致，当前通道数={channels}，"
            f"但拿到的是 shape={arr.shape}。"
        )
    return arr.reshape((1, 1, channels))

def preprocess_input(image):
    # 原始 DeepLab 代码只做 image /= 255.0，这适合 8bit RGB jpg/png。
    #
    # 多光谱 Sentinel-2 tif 通常是 uint16 反射率整数值，例如 0~10000。
    # 如果继续 /255，数值会被放大到几十，和模型期望分布完全不一致。
    #
    # 因此这里按输入数值范围分两条路：
    # 1. max <= 255：
    #    认为是普通 8bit RGB/灰度图，保留旧逻辑 /255。
    # 2. max > 255：
    #    认为是 Sentinel-2 这类多光谱数据，按配置执行：
    #    / reflectance_scale -> 可选 clip -> 可选 mean/std。
    image = image.astype(np.float32, copy=False)

    if image.size == 0:
        return image

    max_value = float(np.max(image))
    if max_value <= 255:
        image /= 255.0
        return image

    reflectance_scale = float(normalization_config.get("reflectance_scale", 10000.0))
    if reflectance_scale <= 0:
        raise ValueError(f"reflectance_scale 必须大于 0，当前值={reflectance_scale}")
    image /= reflectance_scale

    channels = image.shape[2] if image.ndim == 3 else 1

    if normalization_config.get("enable_clip", False):
        clip_min = _reshape_band_vector(normalization_config.get("clip_min"), channels, "clip_min")
        clip_max = _reshape_band_vector(normalization_config.get("clip_max"), channels, "clip_max")
        if np.any(clip_max <= clip_min):
            raise ValueError("clip_max 中的每个值都必须大于 clip_min 对应位置的值。")
        image = np.clip(image, clip_min, clip_max)

    if normalization_config.get("enable_mean_std", False):
        mean = _reshape_band_vector(normalization_config.get("mean"), channels, "mean")
        std = _reshape_band_vector(normalization_config.get("std"), channels, "std")
        if np.any(std <= 0):
            raise ValueError("std 中的每个值都必须大于 0。")
        image = (image - mean) / std

    return image

def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)

def download_weights(backbone, model_dir="./model_data"):
    import os
    from torch.hub import load_state_dict_from_url
    
    download_urls = {
        'mobilenet' : 'https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/mobilenet_v2.pth.tar',
        'xception'  : 'https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/xception_pytorch_imagenet.pth',
    }
    url = download_urls[backbone]
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    load_state_dict_from_url(url, model_dir)
