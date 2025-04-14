

'''
Reference 

https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/vision-background-removal/vision-background-removal.ipynb


使用 U^2-Net 和 OpenVINO™进行图像背景去除


本笔记本演示了使用 U^2-Net 和 OpenVINO 进行图像背景去除

关于 U2-Net 的更多信息，包括源代码和测试数据，请参阅 GitHub 页面和论文：U^2-Net：使用嵌套 U 结构进行显著目标检测的深度探索。

PyTorch U2-Net 模型已转换为 OpenVINO IR 格式。模型源代码在此处提供。



准备
    安装要求
    导入 pytorch 库和 u2-net
    设置
    加载 u2-Net 模型
将 u2 模型转换为 OpenVINO IR
加载并预处理输入图像
选择推理设备
在 OpenVINO IR 模型上执行推理
可视化结果
    添加背景图像


'''

import os
import time
from collections import namedtuple
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import openvino as ov
import torch





from ov_notebook_utils import download_file, load_image, device_widget


dir = './models/ov_artifacts'

if not Path(f"{dir}/u2net.py").exists():
    download_file(
        url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/vision-background-removal/model/u2net.py", directory=dir
    )
from models.ov_artifacts.u2net import U2NET, U2NETP





model_config = namedtuple("ModelConfig", ["name", "url", "model", "model_args"])

u2net_lite = model_config(
    name="u2net_lite",
    url="https://drive.google.com/uc?id=1W8E4FHIlTVstfRkYmNOjbr0VDXTZm0jD",
    model=U2NETP,
    model_args=(),
)

u2net = model_config(
    name="u2net",
    url="https://drive.google.com/uc?id=1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ",
    model=U2NET,
    model_args=(3, 1),
)

u2net_human_seg = model_config(
    name="u2net_human_seg",
    url="https://drive.google.com/uc?id=1m_Kgs91b21gayc2XLW0ou8yugAIadWVP",
    model=U2NET,
    model_args=(3, 1),
)


# Set u2net_model to one of the three configurations listed above.
u2net_model = u2net_lite


# The filenames of the downloaded and converted models.
MODEL_DIR = "models/ov_artifacts"
model_path = Path(MODEL_DIR) / u2net_model.name / Path(u2net_model.name).with_suffix(".pth")


# 加载 U2-Net 模型

# U2-Net 人体分割模型权重存储在 Google Drive 上。如果尚未存在，则会下载。下一个单元格加载模型和预训练权重。

if not model_path.exists():
    import gdown

    os.makedirs(name=model_path.parent, exist_ok=True)
    print("Start downloading model weights file... ")
    with open(model_path, "wb") as model_file:
        gdown.download(url=u2net_model.url, output=model_file)
        print(f"Model weights have been downloaded to {model_path}")



# Load the model.
net = u2net_model.model(*u2net_model.model_args)
net.eval()

# Load the weights.
print(f"Loading model weights from: '{model_path}'")
net.load_state_dict(state_dict=torch.load(model_path, map_location="cpu"))




# 将 PyTorch U2-Net 模型转换为 OpenVINO IR

# 我们使用模型转换 Python API 将 PyTorch 模型转换为 OpenVINO IR 格式。执行以下命令可能需要一段时间。


model_ir = ov.convert_model(net, example_input=torch.zeros((1, 3, 512, 512)), input=([1, 3, 512, 512]))

'''

当 OpenCV 以 BGR 格式读取图像时，OpenVINO IR 模型期望图像为 RGB 格式。因此，将图像转换为 RGB ，
调整大小为 512 x 512 ，并将维度转置为 OpenVINO IR 模型期望的格式。

我们将平均值添加到图像张量中，并用标准差缩放输入。在通过网络传播之前，这被称为输入数据归一化。
平均值和标准差值可以在 U^2-Net 存储库中的 dataloader 文件中找到，并将其乘以 255 以支持像素值为 0-255 的图像。

'''



IMAGE_URI = "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/coco_hollywood.jpg"
IMAGE_NAME = "coco_hollywood.jpg"

input_mean = np.array([123.675, 116.28, 103.53]).reshape(1, 3, 1, 1)
input_scale = np.array([58.395, 57.12, 57.375]).reshape(1, 3, 1, 1)

image = cv2.cvtColor(
    src=load_image(IMAGE_NAME, IMAGE_URI),
    code=cv2.COLOR_BGR2RGB,
)

resized_image = cv2.resize(src=image, dsize=(512, 512))
# Convert the image shape to a shape and a data type expected by the network
# for OpenVINO IR model: (1, 3, 512, 512).
input_image = np.expand_dims(np.transpose(resized_image, (2, 0, 1)), 0)

input_image = (input_image - input_mean) / input_scale

print(f'input_image shape is {input_image.shape}')


device = 'CPU'



# 加载 OpenVINO IR 模型到 OpenVINO 运行时并进行推理。



core = ov.Core()
# Load the network to OpenVINO Runtime.
compiled_model_ir = core.compile_model(model=model_ir, device_name=device)
# Get the names of input and output layers.
input_layer_ir = compiled_model_ir.input(0)
output_layer_ir = compiled_model_ir.output(0)

# Do inference on the input image.
start_time = time.perf_counter()
result = compiled_model_ir([input_image])[output_layer_ir]
end_time = time.perf_counter()
print(f"Inference finished. Inference time: {end_time-start_time:.3f} seconds, " f"FPS: {1/(end_time-start_time):.2f}.")




# Resize the network result to the image shape and round the values
# to 0 (background) and 1 (foreground).
# The network result has (1,1,512,512) shape. The `np.squeeze` function converts this to (512, 512).
# 重新调整结果大小
resized_result = np.rint(cv2.resize(src=np.squeeze(result), dsize=(image.shape[1], image.shape[0]))).astype(np.uint8)

# Create a copy of the image and set all background values to 255 (white).
bg_removed_result = image.copy()
bg_removed_result[resized_result == 0] = 255

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 7))
ax[0].imshow(image)
ax[1].imshow(resized_result, cmap="gray")
ax[2].imshow(bg_removed_result)
for a in ax:
    a.axis("off")


plt.show()








