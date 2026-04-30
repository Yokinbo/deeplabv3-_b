import colorsys
import copy
import time

import cv2
import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from nets.deeplabv3_plus import DeepLab
from utils.utils import cvtColor, preprocess_input, resize_image, show_config
from multispectral_config import (image_ext, in_channels, selected_bands,
                                  trained_model_path, vis_bands)

#训练完之后进行预测与计算miou都需要修改这里
#计算get_miou也需要在这里修改
##模型训练时应该不用改这部分代码，模型进行预测时需要修改下面的三个参数：model_path、backbone和num_classes
#-----------------------------------------------------------------------------------#
#   使用自己训练好的模型预测需要修改3个参数
#   model_path、backbone和num_classes都需要修改！
#   如果出现shape不匹配，一定要注意训练时的model_path、backbone和num_classes的修改
#-----------------------------------------------------------------------------------#
class DeeplabV3(object):
    _defaults = {
        #-------------------------------------------------------------------#
        #   model_path指向logs文件夹下的权值文件
        #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
        #   验证集损失较低不代表miou较高，仅代表该权值在验证集上泛化性能较好。
        #-------------------------------------------------------------------#
        "model_path"        : trained_model_path,
        #----------------------------------------#
        #   所需要区分的类的个数+1
        #----------------------------------------#
        "num_classes"       : 2,
        #----------------------------------------#
        #   所使用的的主干网络：
        #   mobilenet
        #   xception    
        #----------------------------------------#
        "backbone"          : "mobilenet",
        #----------------------------------------#
        #   输入图片的大小
        #----------------------------------------#
        "input_shape"       : [512, 512],
        #----------------------------------------#
        #   多光谱相关配置：
        #   image_ext      影像后缀，例如 .tif / .jpg
        #   selected_bands 实际送进模型的波段，rasterio 使用 1-based 编号
        #   vis_bands      多波段可视化时用于生成 RGB 预览的波段
        #   in_channels    模型第一层输入通道数，必须等于 selected_bands 长度
        #----------------------------------------#
        "image_ext"         : image_ext,
        "selected_bands"    : selected_bands,
        "vis_bands"         : vis_bands,
        "in_channels"       : in_channels,
        #----------------------------------------#
        #   下采样的倍数，一般可选的为8和16
        #   与训练时设置的一样即可
        #----------------------------------------#
        "downsample_factor" : 16,
        #-------------------------------------------------#
        #   mix_type参数用于控制检测结果的可视化方式
        #
        #   mix_type = 0的时候代表原图与生成的图进行混合
        #   mix_type = 1的时候代表仅保留生成的图
        #   mix_type = 2的时候代表仅扣去背景，仅保留原图中的目标
        #-------------------------------------------------#
        "mix_type"          : 0,
        #-------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        #-------------------------------#
        "cuda"              : True,
    }

    #---------------------------------------------------#
    #   初始化Deeplab
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        # 如果用户只显式传了 selected_bands，没有同步传 in_channels，
        # 这里自动按波段数量修正，减少手动配置出错的机会。
        # 若 selected_bands=None，则表示读取 tif 的全部波段，此时保留用户传入的 in_channels。
        if self.selected_bands is not None:
            self.in_channels = len(self.selected_bands)
        #---------------------------------------------------#
        #   画框设置不同的颜色
        #---------------------------------------------------#
        if self.num_classes <= 21:
            self.colors = [ (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128), 
                            (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128), 
                            (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128), 
                            (128, 64, 12)]
        else:
            hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        #---------------------------------------------------#
        #   获得模型
        #---------------------------------------------------#
        self.generate()
        
        # 打印实际生效配置，而不是只打印 _defaults。
        # 这样当外部传入 model_path / backbone / selected_bands 等参数时，
        # 控制台显示的就是当前真正用于推理的配置。
        show_config(**{key: getattr(self, key) for key in self._defaults})
                    
    #---------------------------------------------------#
    #   获得所有的分类
    #---------------------------------------------------#
    def generate(self, onnx=False):
        #-------------------------------#
        #   载入模型与权值
        #-------------------------------#
        self.net = DeepLab(
            num_classes=self.num_classes,
            backbone=self.backbone,
            downsample_factor=self.downsample_factor,
            pretrained=False,
            in_channels=self.in_channels,
        )

        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net    = self.net.eval()
        print('{} model, and classes loaded.'.format(self.model_path))
        if not onnx:
            if self.cuda:
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()

    def read_image(self, image):
        # predict.py 里为了避免 PIL 打开 tif 丢失多波段信息，会直接把路径传进来。
        # 这里统一处理：
        # - tif/tiff 路径：rasterio 读取 selected_bands，返回 HWC numpy
        # - 其它路径：PIL 打开，保持原 RGB 兼容流程
        # - 已经传入的 PIL/numpy：直接返回
        if not isinstance(image, str):
            return image

        if image.lower().endswith((".tif", ".tiff")):
            with rasterio.open(image) as src:
                if self.selected_bands is None:
                    band_indexes = list(range(1, src.count + 1))
                else:
                    band_indexes = self.selected_bands

                if max(band_indexes) > src.count:
                    raise ValueError(
                        f"{image} 只有 {src.count} 个波段，"
                        f"但当前 selected_bands={band_indexes}。"
                    )

                arr = src.read(indexes=band_indexes)
                arr = np.transpose(arr, (1, 2, 0))
            return arr

        return Image.open(image)

    def resize_multiband_image(self, image, size):
        # resize_image 使用 PIL，只适合 RGB。
        # 多波段 numpy 影像需要用 cv2 做等比例缩放，再手动 padding。
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

    def multiband_to_rgb_preview(self, image):
        # 检测结果 mix_type=0/2 需要和原图叠加显示。
        # 但 4/6 波段数组不能直接 Image.blend，所以这里用 vis_bands 生成一张 RGB 预览图。
        if isinstance(image, Image.Image):
            return cvtColor(image)

        image = np.array(image)
        if image.ndim == 2:
            image = np.expand_dims(image, -1)

        vis_band_indexes = self.vis_bands or [1, 2, 3]
        if image.shape[2] >= max(vis_band_indexes):
            idx = [band - 1 for band in vis_band_indexes]
        elif image.shape[2] >= 3:
            idx = [0, 1, 2]
        else:
            idx = [0, 0, 0]

        rgb = image[:, :, idx].astype(np.float32)
        out = np.zeros_like(rgb, dtype=np.uint8)
        for i in range(rgb.shape[2]):
            band = rgb[:, :, i]
            low, high = np.percentile(band, [2, 98])
            if high <= low:
                out[:, :, i] = np.clip(band, 0, 255).astype(np.uint8)
            else:
                out[:, :, i] = np.clip((band - low) / (high - low) * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(out)

    def prepare_input(self, image):
        # 统一把 PIL/RGB 或 tif 多波段影像整理成模型输入：
        # - image_data: [1, C, H, W]
        # - nw/nh: letterbox resize 后的有效区域尺寸
        # - original_h/original_w: 原图尺寸
        # - old_img: 用于结果叠加显示的 RGB 预览图
        image = self.read_image(image)

        if isinstance(image, Image.Image):
            image = cvtColor(image)
            old_img = copy.deepcopy(image)
            image_array = np.array(image, np.float32)
            image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))
            image_data = np.array(image_data, np.float32)
        else:
            image_array = np.array(image, np.float32)
            if image_array.ndim == 2:
                image_array = np.expand_dims(image_array, -1)
            old_img = self.multiband_to_rgb_preview(image_array)
            image_data, nw, nh = self.resize_multiband_image(image_array, (self.input_shape[1], self.input_shape[0]))

        original_h = image_array.shape[0]
        original_w = image_array.shape[1]
        image_data = np.expand_dims(np.transpose(preprocess_input(image_data), (2, 0, 1)), 0)
        return image_data, nw, nh, original_h, original_w, old_img

    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image, count=False, name_classes=None):
        image_data, nw, nh, orininal_h, orininal_w, old_img = self.prepare_input(image)

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
        
        #---------------------------------------------------------#
        #   计数
        #---------------------------------------------------------#
        if count:
            classes_nums        = np.zeros([self.num_classes])
            total_points_num    = orininal_h * orininal_w
            print('-' * 63)
            print("|%25s | %15s | %15s|"%("Key", "Value", "Ratio"))
            print('-' * 63)
            for i in range(self.num_classes):
                num     = np.sum(pr == i)
                ratio   = num / total_points_num * 100
                if num > 0:
                    print("|%25s | %15s | %14.2f%%|"%(str(name_classes[i]), str(num), ratio))
                    print('-' * 63)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)
    
        if self.mix_type == 0:
            # seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))
            # for c in range(self.num_classes):
            #     seg_img[:, :, 0] += ((pr[:, :] == c ) * self.colors[c][0]).astype('uint8')
            #     seg_img[:, :, 1] += ((pr[:, :] == c ) * self.colors[c][1]).astype('uint8')
            #     seg_img[:, :, 2] += ((pr[:, :] == c ) * self.colors[c][2]).astype('uint8')
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
            #------------------------------------------------#
            #   将新图片转换成Image的形式
            #------------------------------------------------#
            image   = Image.fromarray(np.uint8(seg_img))
            #------------------------------------------------#
            #   将新图与原图及进行混合
            #------------------------------------------------#
            image   = Image.blend(old_img, image, 0.7)

        elif self.mix_type == 1:
            # seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))
            # for c in range(self.num_classes):
            #     seg_img[:, :, 0] += ((pr[:, :] == c ) * self.colors[c][0]).astype('uint8')
            #     seg_img[:, :, 1] += ((pr[:, :] == c ) * self.colors[c][1]).astype('uint8')
            #     seg_img[:, :, 2] += ((pr[:, :] == c ) * self.colors[c][2]).astype('uint8')
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
            #------------------------------------------------#
            #   将新图片转换成Image的形式
            #------------------------------------------------#
            image   = Image.fromarray(np.uint8(seg_img))

        elif self.mix_type == 2:
            seg_img = (np.expand_dims(pr != 0, -1) * np.array(old_img, np.float32)).astype('uint8')
            #------------------------------------------------#
            #   将新图片转换成Image的形式
            #------------------------------------------------#
            image = Image.fromarray(np.uint8(seg_img))
        
        return image

    def get_FPS(self, image, test_interval):
        image_data, nw, nh, _, _, _ = self.prepare_input(image)

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
            pr = F.softmax(pr.permute(1,2,0),dim = -1).cpu().numpy().argmax(axis=-1)
            #--------------------------------------#
            #   将灰条部分截取掉
            #--------------------------------------#
            pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                    int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]

        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                #---------------------------------------------------#
                #   图片传入网络进行预测
                #---------------------------------------------------#
                pr = self.net(images)[0]
                #---------------------------------------------------#
                #   取出每一个像素点的种类
                #---------------------------------------------------#
                pr = F.softmax(pr.permute(1,2,0),dim = -1).cpu().numpy().argmax(axis=-1)
                #--------------------------------------#
                #   将灰条部分截取掉
                #--------------------------------------#
                pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                        int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    def convert_to_onnx(self, simplify, model_path):
        import onnx
        self.generate(onnx=True)

        im                  = torch.zeros(1, self.in_channels, *self.input_shape).to('cpu')  # BCHW
        input_layer_names   = ["images"]
        output_layer_names  = ["output"]
        
        # Export the model
        print(f'Starting export with onnx {onnx.__version__}.')
        torch.onnx.export(self.net,
                        im,
                        f               = model_path,
                        verbose         = False,
                        opset_version   = 12,
                        training        = torch.onnx.TrainingMode.EVAL,
                        do_constant_folding = True,
                        input_names     = input_layer_names,
                        output_names    = output_layer_names,
                        dynamic_axes    = None)

        # Checks
        model_onnx = onnx.load(model_path)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model

        # Simplify onnx
        if simplify:
            import onnxsim
            print(f'Simplifying with onnx-simplifier {onnxsim.__version__}.')
            model_onnx, check = onnxsim.simplify(
                model_onnx,
                dynamic_input_shape=False,
                input_shapes=None)
            assert check, 'assert check failed'
            onnx.save(model_onnx, model_path)

        print('Onnx model save as {}'.format(model_path))
    
    def get_miou_png(self, image):
        image_data, nw, nh, orininal_h, orininal_w, _ = self.prepare_input(image)

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
