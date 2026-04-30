#--------------------------------------------#
#   该部分代码用于看网络结构
#--------------------------------------------#
import torch
from thop import clever_format, profile
from torchsummary import summary

from nets.deeplabv3_plus import DeepLab
from multispectral_config import band_mode, in_channels, selected_bands

if __name__ == "__main__":
    input_shape     = [512, 512]
    num_classes     = 2
    backbone        = 'mobilenet'
    
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 多光谱模型的第一层输入通道数由 multispectral_config.py 统一控制。
    # 当前默认 6band 时，in_channels=6；如果这里仍然写死 3，
    # summary / FLOPs 统计会和真实模型输入不一致。
    model   = DeepLab(
        num_classes=num_classes,
        backbone=backbone,
        downsample_factor=16,
        pretrained=False,
        in_channels=in_channels,
    ).to(device)
    summary(model, (in_channels, input_shape[0], input_shape[1]))
    
    dummy_input     = torch.randn(1, in_channels, input_shape[0], input_shape[1]).to(device)
    flops, params   = profile(model.to(device), (dummy_input, ), verbose=False)
    #--------------------------------------------------------#
    #   flops * 2是因为profile没有将卷积作为两个operations
    #   有些论文将卷积算乘法、加法两个operations。此时乘2
    #   有些论文只考虑乘法的运算次数，忽略加法。此时不乘2
    #   本代码选择乘2，参考YOLOX。
    #--------------------------------------------------------#
    flops           = flops * 2
    flops, params   = clever_format([flops, params], "%.3f")
    print("band_mode: %s" % band_mode)
    print("selected_bands: %s" % selected_bands)
    print("in_channels: %s" % in_channels)
    print('Total GFLOPS: %s' % (flops))
    print('Total params: %s' % (params))
