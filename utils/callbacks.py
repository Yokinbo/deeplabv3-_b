import os

import matplotlib
import torch
import torch.nn.functional as F

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import scipy.signal

import cv2
import shutil
import numpy as np
import rasterio

from PIL import Image
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from .utils import cvtColor, preprocess_input, resize_image
from .utils_metrics import compute_mIoU


class LossHistory():
    def __init__(self, log_dir, model, input_shape, in_channels=3):
        self.log_dir    = log_dir
        self.losses     = []
        self.val_loss   = []
        
        os.makedirs(self.log_dir)
        self.writer     = SummaryWriter(self.log_dir)
        try:
            # TensorBoard add_graph 会真的跑一次模型。
            # 多光谱训练时模型第一层可能是 4/6 通道，所以 dummy input
            # 也必须跟随当前 in_channels，而不能继续固定写 3。
            dummy_input     = torch.randn(2, in_channels, input_shape[0], input_shape[1])
            self.writer.add_graph(model, dummy_input)
        except:
            pass

    def append_loss(self, epoch, loss, val_loss):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.losses.append(loss)
        self.val_loss.append(val_loss)

        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
            f.write(str(val_loss))
            f.write("\n")

        self.writer.add_scalar('loss', loss, epoch)
        self.writer.add_scalar('val_loss', val_loss, epoch)
        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth = 2, label='train loss')
        plt.plot(iters, self.val_loss, 'coral', linewidth = 2, label='val loss')
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15
            
            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle = '--', linewidth = 2, label='smooth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle = '--', linewidth = 2, label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))

        plt.cla()
        plt.close("all")

class EvalCallback():
    def __init__(self, net, input_shape, num_classes, image_ids, dataset_path, log_dir, cuda, \
            miou_out_path=".temp_miou_out", eval_flag=True, period=1, image_ext=".jpg", selected_bands=None):
        super(EvalCallback, self).__init__()
        
        self.net                = net
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.image_ids          = image_ids
        self.dataset_path       = dataset_path
        self.log_dir            = log_dir
        self.cuda               = cuda
        self.miou_out_path      = miou_out_path
        self.eval_flag          = eval_flag
        self.period             = period
        # 验证阶段必须和训练阶段读取同样的影像后缀和波段。
        # 否则会出现训练用 6 波段 tif，mIoU 却仍然读 jpg/RGB 的不一致。
        self.image_ext          = image_ext
        self.selected_bands     = selected_bands
        
        self.image_ids          = [image_id.split()[0] for image_id in image_ids]
        self.mious      = [0]
        self.epoches    = [0]
        if self.eval_flag:
            with open(os.path.join(self.log_dir, "epoch_miou.txt"), 'a') as f:
                f.write(str(0))
                f.write("\n")

    def read_eval_image(self, image_path):
        # tif/tiff 使用 rasterio 读取多波段；其它格式继续使用 PIL。
        if self.image_ext.lower() in [".tif", ".tiff"]:
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
        return Image.open(image_path)

    def resize_multiband_image(self, image, size):
        # resize_image 只适合 PIL/RGB。
        # 多波段 numpy 影像需要自己做等比例缩放 + padding。
        ih, iw  = image.shape[:2]
        w, h    = size

        scale   = min(w / iw, h / ih)
        nw      = int(iw * scale)
        nh      = int(ih * scale)

        image   = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        if image.ndim == 2:
            image = np.expand_dims(image, -1)

        channels = image.shape[2]
        new_image = np.zeros((h, w, channels), dtype=image.dtype)
        new_image[(h - nh) // 2:(h - nh) // 2 + nh, (w - nw) // 2:(w - nw) // 2 + nw, :] = image
        return new_image, nw, nh

    def get_miou_png(self, image):
        #---------------------------------------------------------#
        #   RGB/PIL 图像走原来的 cvtColor + resize_image 流程。
        #   多光谱 tif 已经在 read_eval_image 中读成 numpy 多通道数组，
        #   这里会跳过 RGB 转换，直接做多波段 resize 和标准化。
        #---------------------------------------------------------#
        # RGB/PIL 保留原逻辑；多波段 tif 直接按 numpy 多通道处理。
        if isinstance(image, Image.Image):
            image       = cvtColor(image)
            image_array = np.array(image, np.float32)
            image_data, nw, nh  = resize_image(image, (self.input_shape[1],self.input_shape[0]))
            image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)
        else:
            image_array = np.array(image, np.float32)
            if image_array.ndim == 2:
                image_array = np.expand_dims(image_array, -1)
            image_data, nw, nh = self.resize_multiband_image(image_array, (self.input_shape[1], self.input_shape[0]))
            image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        orininal_h  = image_array.shape[0]
        orininal_w  = image_array.shape[1]

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
                
            #---------------------------------------------------#
            #   图片传入网络进行预测
            #---------------------------------------------------#
            pr = self.net(images)[0]
            #---------------------------------------------------#
            #   取出每一个像素点的种类
            #---------------------------------------------------#
            pr = F.softmax(pr.permute(1,2,0),dim = -1).cpu().numpy()
            #--------------------------------------#
            #   将灰条部分截取掉
            #--------------------------------------#
            pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                    int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
            #---------------------------------------------------#
            #   进行图片的resize
            #---------------------------------------------------#
            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation = cv2.INTER_LINEAR)
            #---------------------------------------------------#
            #   取出每一个像素点的种类
            #---------------------------------------------------#
            pr = pr.argmax(axis=-1)
    
        image = Image.fromarray(np.uint8(pr))
        return image
    
    def on_epoch_end(self, epoch, model_eval):
        if epoch % self.period == 0 and self.eval_flag:
            self.net    = model_eval
            gt_dir      = os.path.join(self.dataset_path, "VOC2007/SegmentationClass/")
            pred_dir    = os.path.join(self.miou_out_path, 'detection-results')
            if not os.path.exists(self.miou_out_path):
                os.makedirs(self.miou_out_path)
            if not os.path.exists(pred_dir):
                os.makedirs(pred_dir)
            print("Get miou.")
            for image_id in tqdm(self.image_ids):
                #-------------------------------#
                #   从文件中读取图像
                #-------------------------------#
                image_path  = os.path.join(self.dataset_path, "VOC2007/JPEGImages", image_id + self.image_ext)
                image       = self.read_eval_image(image_path)
                #------------------------------#
                #   获得预测txt
                #------------------------------#
                image       = self.get_miou_png(image)
                image.save(os.path.join(pred_dir, image_id + ".png"))
                        
            print("Calculate miou.")
            _, IoUs, _, _ = compute_mIoU(gt_dir, pred_dir, self.image_ids, self.num_classes, None)  # 执行计算mIoU的函数
            temp_miou = np.nanmean(IoUs) * 100

            self.mious.append(temp_miou)
            self.epoches.append(epoch)

            with open(os.path.join(self.log_dir, "epoch_miou.txt"), 'a') as f:
                f.write(str(temp_miou))
                f.write("\n")
            
            plt.figure()
            plt.plot(self.epoches, self.mious, 'red', linewidth = 2, label='train miou')

            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('Miou')
            plt.title('A Miou Curve')
            plt.legend(loc="upper right")

            plt.savefig(os.path.join(self.log_dir, "epoch_miou.png"))
            plt.cla()
            plt.close("all")

            print("Get miou done.")
            shutil.rmtree(self.miou_out_path)
