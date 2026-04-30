# DeepLabV3+ 多光谱使用说明

这个仓库已经在 `multisdeeplab3` 分支上改造成支持多光谱 `.tif` 训练、验证、推理和 mIoU 评估的版本。

当前默认配置和 `E:/UNet/UNet_b` 的 `myimprove` 分支保持一致：使用 Sentinel-2 六波段输入。

## 1. 当前支持的内容

- VOC 目录结构下训练语义分割模型
- `rgb / 4band / 6band` 三种输入模式切换
- DeepLabV3+ 的 `mobilenet` 和 `xception` backbone 多通道输入
- `.tif` 多波段 patch 读取
- 按指定波段读取，例如 `[1, 2, 3, 4, 5, 6]`
- 训练时 mIoU 回调读取多光谱影像
- `predict.py` 多光谱 patch 推理
- `多光谱get_miou.py` 独立 mIoU 评估
- `summary.py` 按当前输入通道统计模型结构和 FLOPs

## 2. 数据目录

训练仍然使用原来的 VOC 格式：

```text
VOCdevkit/
  VOC2007/
    JPEGImages/
      sample_001.tif
      sample_002.tif
      ...
    SegmentationClass/
      sample_001.png
      sample_002.png
      ...
    ImageSets/
      Segmentation/
        train.txt
        val.txt
        trainval.txt
        test.txt
```

注意：

- `JPEGImages` 这个文件夹名字保留不变，但里面现在可以放 `.tif`。
- 标签仍然放在 `SegmentationClass`，格式是单通道 `.png`。
- 影像和标签必须同名，只是后缀不同。
- 标签像素值需要是类别 id，例如背景为 `0`，光伏为 `1`。

## 3. 统一配置文件

多光谱相关配置集中在：

- [multispectral_config.py](multispectral_config.py)

当前默认：

```python
image_ext = ".tif"
band_mode = "6band"
selected_bands = [1, 2, 3, 4, 5, 6]
in_channels = 6
```

当前假设你的 6 波段 tif 内部顺序是：

```text
1 -> B2
2 -> B3
3 -> B4
4 -> B8
5 -> B11
6 -> B12
```

因此三种模式对应：

```python
"rgb"   -> [3, 2, 1]          # B4, B3, B2
"4band" -> [1, 2, 3, 4]       # B2, B3, B4, B8
"6band" -> [1, 2, 3, 4, 5, 6] # B2, B3, B4, B8, B11, B12
```

如果要切换实验，只优先改：

```python
band_mode = "rgb"
# 或
band_mode = "4band"
# 或
band_mode = "6band"
```

`selected_bands` 和 `in_channels` 会自动跟着变化。

## 4. 标准化配置

`multispectral_config.py` 里还有：

```python
normalization_configs
normalization_config
```

当前预处理链路是：

```text
普通 8bit RGB 图像 -> /255
多光谱 uint16 tif -> /10000 -> clip -> mean/std
```

`rgb / 4band / 6band` 各自有独立的 `clip_min / clip_max / mean / std`。

这些参数已经先沿用 `UNet_b/myimprove` 当前配置，方便两个模型做同款对比。

## 5. 训练

训练脚本仍然是：

```bash
python train.py
```

训练时会从 `multispectral_config.py` 读取：

- `band_mode`
- `image_ext`
- `selected_bands`
- `in_channels`
- `normalization_config`

模型构建时会使用：

```python
DeepLab(..., in_channels=in_channels)
```

训练输出目录会按波段模式自动分开：

```text
logs/rgb/
logs/4band/
logs/6band/
```

当前默认 `band_mode = "6band"`，因此权重会保存到：

```text
logs/6band/
```

## 6. 推理

单张或文件夹预测使用：

```bash
python predict.py
```

`predict.py` 会自动读取：

```python
trained_model_path
image_ext
selected_bands
in_channels
```

如果输入是 `.tif`，脚本不会用 PIL 打开，而是直接把路径传给 `deeplab.py`，由内部用 `rasterio` 读取多波段数据。

文件夹预测时，输出统一保存为 `.png`，例如：

```text
input:  sample_001.tif
output: sample_001.png
```

## 7. 独立 mIoU 评估

使用：

```bash
python 多光谱get_miou.py
```

默认评估：

```text
VOCdevkit/VOC2007/ImageSets/Segmentation/val.txt
```

输出目录：

```text
miou_out/
  detection-results/
  miou_summary.txt
```

脚本里的：

```python
miou_mode = 0
```

含义：

```text
0 -> 生成预测结果 + 计算 mIoU
1 -> 只生成预测结果
2 -> 只计算 mIoU
```

如果要评估测试集，可以把脚本里的：

```python
image_set = "val.txt"
```

改成：

```python
image_set = "test.txt"
```

## 8. 模型结构和 FLOPs

使用：

```bash
python summary.py
```

`summary.py` 会按当前 `in_channels` 创建 dummy input。

例如当前 `6band` 时，会统计：

```text
[1, 6, 512, 512]
```

而不是原来的：

```text
[1, 3, 512, 512]
```

## 9. 当前关键改动文件

- [multispectral_config.py](multispectral_config.py)
- [train.py](train.py)
- [deeplab.py](deeplab.py)
- [predict.py](predict.py)
- [多光谱get_miou.py](多光谱get_miou.py)
- [summary.py](summary.py)
- [utils/dataloader.py](utils/dataloader.py)
- [utils/utils.py](utils/utils.py)
- [utils/callbacks.py](utils/callbacks.py)
- [nets/deeplabv3_plus.py](nets/deeplabv3_plus.py)
- [nets/mobilenetv2.py](nets/mobilenetv2.py)
- [nets/xception.py](nets/xception.py)

## 10. 常见注意点

1. 如果出现模型第一层 shape mismatch，优先检查：

```python
band_mode
selected_bands
in_channels
model_path
```

2. 如果输入影像只有 3 个波段，但配置是 `6band`，会报波段不足。

3. 如果使用 `.jpg` 老数据，需要把：

```python
image_ext = ".jpg"
band_mode = "rgb"
```

并确认模型权重也是 RGB 模式训练出来的。

4. 多光谱可视化叠加使用：

```python
vis_bands = [3, 2, 1]
```

也就是用 `[B4, B3, B2]` 生成真彩色预览图。

5. 当前多光谱增强先保留几何增强，不对 4/6 波段做 HSV 色彩增强，因为 HSV 会破坏遥感波段的物理意义。
