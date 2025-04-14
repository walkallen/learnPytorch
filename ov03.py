

'''
Reference 


https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/vision-monodepth/vision-monodepth.ipynb

Monodepth Estimation with OpenVINO
使用 OpenVINO 进行单目深度估计

本教程演示了使用 MidasNet 在 OpenVINO 中进行单目深度估计。模型信息可在此处找到。
https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/midasnet/README.md




什么是 Monodepth？


单目深度估计是使用单张图像估计场景深度的任务。它在机器人、3D 重建、医学成像和自主系统等领域具有许多潜在应用。
本教程使用由 Embodied AI Foundation 开发的 MiDaS 神经网络模型。请参阅以下研究论文以了解更多信息。


R. Ranftl, K. Lasinger, D. Hafner, K. Schindler 和 V. Koltun, "迈向鲁棒的单目深度估计：混合数据集实现零样本跨数据集迁移," 在 IEEE 传输模式分析与机器智能杂志上，doi: 10.1109/TPAMI.2020.3019967 .


准备
    安装要求
    导入
    下载模型
函数
选择推理设备
加载模型

单目深度图像
    加载，调整大小和重塑输入图像
    在图像上执行推理
    显示单目深度图像

单目深度处理视频
    视频设置
    加载视频
    在视频中执行推理并创建单目深度视频
    显示 Monodepth 视频


'''




import time

import cv2
import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np



import openvino as ov

from pathlib import Path


from ov_notebook_utils import download_file, load_image, device_widget




'''
下载模型

The model is in the OpenVINO Intermediate Representation (IR) format.
该模型处于 OpenVINO 中间表示（IR）格式。



'''


model_folder = Path("models/ov_artifacts")

ir_model_url = "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/models/depth-estimation-midas/FP32/"
ir_model_name_xml = "MiDaS_small.xml"
ir_model_name_bin = "MiDaS_small.bin"

download_file(ir_model_url + ir_model_name_xml, filename=ir_model_name_xml, directory=model_folder)
download_file(ir_model_url + ir_model_name_bin, filename=ir_model_name_bin, directory=model_folder)

model_xml_path = model_folder / ir_model_name_xml




def normalize_minmax(data):
    """Normalizes the values in `data` between 0 and 1"""
    return (data - data.min()) / (data.max() - data.min())


def convert_result_to_image(result, colormap="viridis"):
    """
    Convert network result of floating point numbers to an RGB image with
    integer values from 0-255 by applying a colormap.

    `result` is expected to be a single network result in 1,H,W shape
    `colormap` is a matplotlib colormap.
    See https://matplotlib.org/stable/tutorials/colors/colormaps.html
    """
    cmap = matplotlib.cm.get_cmap(colormap)
    result = result.squeeze(0)
    result = normalize_minmax(result)
    result = cmap(result)[:, :, :3] * 255
    result = result.astype(np.uint8)
    return result


def to_rgb(image_data) -> np.ndarray:
    """
    Convert image_data from BGR to RGB
    """
    return cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)


device = 'CPU'


'''

加载模型

加载 OpenVINO Runtime 中的模型，使用 core.read_model ，并针对指定设备编译 core.compile_model 。
获取模型的输入和输出键以及期望的输入形状。


'''



import openvino.properties as props


'''
使用缓存的好处：

当 OpenVINO 首次将一个模型编译到特定的设备上时，这个过程可能需要一些时间。
通过设置缓存目录，OpenVINO 可以将编译结果保存在该目录下。在后续的运行中，
如果再次加载并编译相同的模型到相同的设备上，OpenVINO 可以直接从缓存中加载已编译的版本，
从而大大加快模型的加载速度，节省编译时间。这对于需要多次运行相同模型的应用场景非常有用。

'''

# Create cache folder
cache_folder = Path("cache")
cache_folder.mkdir(exist_ok=True)

core = ov.Core()
core.set_property({props.cache_dir(): cache_folder})
model = core.read_model(model_xml_path)
compiled_model = core.compile_model(model=model, device_name = device)

input_key = compiled_model.input(0)
output_key = compiled_model.output(0)

print(f'input  shape is {input_key.shape}')
print(f'output shape is {output_key.shape}')

network_input_shape = list(input_key.shape)
network_image_height, network_image_width = network_input_shape[2:]



'''
单目深度估计

加载、调整大小和重塑输入图像


输入图像使用 OpenCV 读取，调整到网络输入大小，并重塑为(N,C,H,W)（N=图像数量，C=通道数，H=高度，W=宽度）。


'''


print("\n======================= 处理输入图片 \n")


IMAGE_URL = "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/coco_bike.jpg"
IMAGE_NAME = "coco_bike.jpg"


image = load_image(IMAGE_NAME, IMAGE_URL)

print(f'image shape is {image.shape}')

# Resize to input shape for network.
resized_image = cv2.resize(src=image, dsize=(network_image_height, network_image_width))

print(f'resized_image shape is {resized_image.shape}')


# Reshape the image to network input shape NCHW.
input_image = np.expand_dims(np.transpose(resized_image, (2, 0, 1)), 0)


print(f'input_image shape is {input_image.shape}')



print("\n======================= 对图像进行推理 \n")

print(f'resize target shape is {image.shape[:2][::-1]}')


# 进行推理，将结果转换为图像，并将其调整到原始图像大小。

result = compiled_model([input_image])[output_key]

# Convert the network result of disparity map to an image that shows
# distance as colors.
result_image = convert_result_to_image(result=result)

print(f'result_image shape is {result_image.shape}')

# Resize back to original image shape. The `cv2.resize` function expects shape
# in (width, height), [::-1] reverses the (height, width) shape to match this.
result_image = cv2.resize(result_image, image.shape[:2][::-1])




fig, ax = plt.subplots(1, 2, figsize=(20, 15))
ax[0].imshow(to_rgb(image))
ax[1].imshow(result_image)
plt.show()


