import os

import cv2
import numpy as np
import torch
import rasterio
from PIL import Image
from torch.utils.data.dataset import Dataset

from utils.utils import cvtColor, preprocess_input


class DeeplabDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, train, dataset_path, image_ext=".jpg", selected_bands=None):
        super(DeeplabDataset, self).__init__()
        self.annotation_lines   = annotation_lines
        self.length             = len(annotation_lines)
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.train              = train
        self.dataset_path       = dataset_path
        # 多光谱改造：
        # 1. image_ext 控制训练影像后缀，既可以是原来的 .jpg，也可以是 .tif。
        # 2. selected_bands 控制 tif 实际读取哪些波段，例如 [1,2,3,4,5,6]。
        #    这里使用 1-based 波段编号，和 rasterio / 遥感软件习惯一致。
        self.image_ext          = image_ext
        self.selected_bands     = selected_bands

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        annotation_line = self.annotation_lines[index]
        name            = annotation_line.split()[0]

        #-------------------------------#
        #   从文件中读取图像
        #-------------------------------#
        image_path  = os.path.join(os.path.join(self.dataset_path, "VOC2007/JPEGImages"), name + self.image_ext)
        if self.image_ext.lower() in [".tif", ".tiff"]:
            jpg     = self.read_tif(image_path)
        else:
            jpg     = Image.open(image_path)
        png         = Image.open(os.path.join(os.path.join(self.dataset_path, "VOC2007/SegmentationClass"), name + ".png"))
        #-------------------------------#
        #   数据增强
        #-------------------------------#
        jpg, png    = self.get_random_data(jpg, png, self.input_shape, random = self.train)

        # get_random_data 之后：
        # - RGB 旧流程可能返回 PIL.Image 或 HWC numpy
        # - 多光谱 tif 流程返回 HWC numpy
        # 这里统一转成 HWC numpy，再交给 preprocess_input 做标准化，最后转 CHW。
        jpg         = np.array(jpg, np.float64)
        if len(np.shape(jpg)) == 2:
            jpg = np.expand_dims(jpg, -1)
        jpg         = np.transpose(preprocess_input(jpg), [2,0,1])
        png         = np.array(png)
        png[png >= self.num_classes] = self.num_classes
        #-------------------------------------------------------#
        #   转化成one_hot的形式
        #   在这里需要+1是因为voc数据集有些标签具有白边部分
        #   我们需要将白边部分进行忽略，+1的目的是方便忽略。
        #-------------------------------------------------------#
        seg_labels  = np.eye(self.num_classes + 1)[png.reshape([-1])]
        seg_labels  = seg_labels.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes + 1))

        return jpg, png, seg_labels

    def read_tif(self, image_path):
        # 使用 rasterio 读取多波段 tif。
        # rasterio 读出来默认是 (C, H, W)，而 OpenCV / numpy 增强习惯使用 (H, W, C)，
        # 所以这里读完后立刻转成 HWC，后面的增强逻辑会更统一。
        with rasterio.open(image_path) as src:
            if self.selected_bands is None:
                band_indexes = list(range(1, src.count + 1))
            else:
                band_indexes = self.selected_bands

            if max(band_indexes) > src.count:
                raise ValueError(
                    f"{image_path} 只有 {src.count} 个波段，"
                    f"但当前 selected_bands={band_indexes}。"
                )

            image = src.read(indexes=band_indexes)
            image = np.transpose(image, (1, 2, 0))
        return image

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image, label, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.3, random=True):
        # 原始 DeepLab 默认把所有输入都转成 RGB。
        # 多光谱 tif 不能这样做，否则 4/6 波段信息会被压成 3 通道。
        # 因此只有 PIL 图像走 cvtColor；numpy 多波段图像保持原通道数。
        if isinstance(image, Image.Image):
            image = cvtColor(image)
        label   = Image.fromarray(np.array(label))
        #------------------------------#
        #   获得图像的高宽与目标高宽
        #------------------------------#
        if isinstance(image, Image.Image):
            iw, ih = image.size
        else:
            ih, iw = image.shape[:2]
        h, w    = input_shape

        if not random:
            if isinstance(image, Image.Image):
                iw, ih = image.size
            else:
                ih, iw = image.shape[:2]
            scale   = min(w/iw, h/ih)
            nw      = int(iw*scale)
            nh      = int(ih*scale)

            if isinstance(image, Image.Image):
                image       = image.resize((nw,nh), Image.BICUBIC)
                new_image   = Image.new('RGB', [w, h], (128,128,128))
                new_image.paste(image, ((w-nw)//2, (h-nh)//2))
            else:
                image       = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
                if image.ndim == 2:
                    image = np.expand_dims(image, -1)
                channels    = image.shape[2]
                new_image   = np.full((h, w, channels), 128, dtype=image.dtype)
                new_image[(h-nh)//2:(h-nh)//2 + nh, (w-nw)//2:(w-nw)//2 + nw, :] = image

            label       = label.resize((nw,nh), Image.NEAREST)
            new_label   = Image.new('L', [w, h], (0))
            new_label.paste(label, ((w-nw)//2, (h-nh)//2))
            return new_image, new_label

        #------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        #------------------------------------------#
        new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        scale = self.rand(0.25, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        if isinstance(image, Image.Image):
            image = image.resize((nw,nh), Image.BICUBIC)
        else:
            image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
            if image.ndim == 2:
                image = np.expand_dims(image, -1)
        label = label.resize((nw,nh), Image.NEAREST)
        
        #------------------------------------------#
        #   翻转图像
        #------------------------------------------#
        flip = self.rand()<.5
        if flip: 
            if isinstance(image, Image.Image):
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            else:
                image = np.ascontiguousarray(image[:, ::-1, :])
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
        
        #------------------------------------------#
        #   将图像多余的部分加上灰条
        #------------------------------------------#
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        if isinstance(image, Image.Image):
            new_image = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image = new_image
        else:
            channels = image.shape[2]
            new_image = np.full((h, w, channels), 128, dtype=image.dtype)
            # PIL 的 paste 可以自动处理图片比画布大、dx/dy 为负等情况。
            # numpy 直接赋值不会自动裁剪，所以这里手动计算源图和目标画布的交集。
            x1 = max(dx, 0)
            y1 = max(dy, 0)
            x2 = min(dx + nw, w)
            y2 = min(dy + nh, h)

            src_x1 = max(-dx, 0)
            src_y1 = max(-dy, 0)
            src_x2 = src_x1 + (x2 - x1)
            src_y2 = src_y1 + (y2 - y1)

            if x2 > x1 and y2 > y1:
                new_image[y1:y2, x1:x2, :] = image[src_y1:src_y2, src_x1:src_x2, :]
            image = new_image
        new_label = Image.new('L', (w,h), (0))
        new_label.paste(label, (dx, dy))
        label = new_label

        image_data      = np.array(image)

        #------------------------------------------#
        #   高斯模糊
        #------------------------------------------#
        blur = self.rand() < 0.25
        if blur and image_data.ndim == 3 and image_data.shape[2] == 3: 
            image_data = cv2.GaussianBlur(image_data, (5, 5), 0)

        #------------------------------------------#
        #   旋转
        #------------------------------------------#
        rotate = self.rand() < 0.25
        if rotate: 
            center      = (w // 2, h // 2)
            rotation    = np.random.randint(-10, 11)
            M           = cv2.getRotationMatrix2D(center, -rotation, scale=1)
            if image_data.ndim == 3 and image_data.shape[2] > 4:
                # OpenCV 的多通道 borderValue 对 5/6 通道数组不够友好。
                # 这里逐波段旋转再堆回去，保证 6band 数据也能稳定增强。
                rotated_bands = [
                    cv2.warpAffine(
                        image_data[:, :, c],
                        M,
                        (w, h),
                        flags=cv2.INTER_CUBIC,
                        borderValue=128,
                    )
                    for c in range(image_data.shape[2])
                ]
                image_data = np.stack(rotated_bands, axis=-1)
            else:
                if image_data.ndim == 3:
                    border_value = tuple([128] * image_data.shape[2])
                else:
                    border_value = 128
                image_data  = cv2.warpAffine(image_data, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=border_value)
            if image_data.ndim == 2:
                image_data = np.expand_dims(image_data, -1)
            label       = cv2.warpAffine(np.array(label, np.uint8), M, (w, h), flags=cv2.INTER_NEAREST, borderValue=(0))

        #---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        #---------------------------------#
        # HSV 色彩增强只适合普通 RGB。
        # 对 4/6 波段遥感影像，强行转 HSV 会破坏波段物理意义，
        # 所以多光谱输入先只保留缩放、翻转、旋转等几何增强。
        if image_data.ndim == 3 and image_data.shape[2] == 3:
            image_data = image_data.astype(np.uint8)
            r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
            #---------------------------------#
            #   将图像转到HSV上
            #---------------------------------#
            hue, sat, val   = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
            dtype           = image_data.dtype
            #---------------------------------#
            #   应用变换
            #---------------------------------#
            x       = np.arange(0, 256, dtype=r.dtype)
            lut_hue = ((x * r[0]) % 180).astype(dtype)
            lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
            lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

            image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
            image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)
        
        return image_data, label


# DataLoader中collate_fn使用
def deeplab_dataset_collate(batch):
    images      = []
    pngs        = []
    seg_labels  = []
    for img, png, labels in batch:
        images.append(img)
        pngs.append(png)
        seg_labels.append(labels)
    images      = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    pngs        = torch.from_numpy(np.array(pngs)).long()
    seg_labels  = torch.from_numpy(np.array(seg_labels)).type(torch.FloatTensor)
    return images, pngs, seg_labels
