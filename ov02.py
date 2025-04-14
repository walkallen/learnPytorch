

'''
Reference 


https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/pytorch-to-openvino/pytorch-onnx-to-openvino.ipynb




将 PyTorch 模型转换为 ONNX 和 OpenVINO™ IR

本教程逐步演示了如何使用 OpenVINO Runtime 在 PyTorch 语义分割模型上进行推理。


首先，将 PyTorch 模型导出为 ONNX 格式，然后转换为 OpenVINO IR。然后分别将相应的 ONNX 和 OpenVINO IR 模型加载到 OpenVINO Runtime 中，以展示模型预测。在本教程中，我们将使用 LR-ASPP 模型和 MobileNetV3 骨干网络。


根据论文《Searching for MobileNetV3》，LR-ASPP（轻量级缩减空洞空间金字塔池化）具有轻量级且高效的分割解码器架构。下方的图示展示了模型架构：


该模型在 MS COCO 数据集上进行了预训练。与训练所有 80 个类别不同，分割模型仅在 PASCAL VOC 数据集的 20 个类别上进行了训练：
背景*、*飞机*、*自行车*、*鸟*、*船*、*瓶子*、*公共汽车*、*汽车*、*猫*、*椅子*、*牛*、*餐桌*、*狗*、*马*、*摩托车*、*人*、
*盆栽植物*、*羊*、*沙发*、*火车*、*电视显示器*


Table of contents:  目录：
Preparation  准备
    Imports      导入
    Settings     设置
    Load Model   加载模型
    
ONNX 模型转换
    将 PyTorch 模型转换为 ONNX
    将 ONNX 模型转换为 OpenVINO IR 格式


Show Results  显示结果

    加载并预处理输入图像

    加载 OpenVINO IR 网络并在 ONNX 模型上运行推理

1. OpenVINO 运行时中的 ONNX 模型
Select inference device  选择推理设备

2. OpenVINO IR 模型在 OpenVINO 运行时中
Select inference device  选择推理设备


PyTorch Comparison  PyTorch 比较
Performance Comparison  性能比较
References  参考文献列表


'''







import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt



from torchvision.models.segmentation import (
    lraspp_mobilenet_v3_large,
    LRASPP_MobileNet_V3_Large_Weights,
)


import openvino as ov

#from ov_notebook_utils import segmentation_map_to_image, viz_result_image, SegmentationMap, Label, download_file, device_widget


from ov_notebook_utils import   download_file
from ov_notebook_utils import   Label, viz_result_image, SegmentationMap, segmentation_map_to_image


# 设置模型的名称，然后定义网络在推理过程中使用的图像的宽度和高度。根据输入转换函数，该模型在高度为 520 像素、宽度为 780 像素的图像上进行了预训练。




IMAGE_WIDTH = 780
IMAGE_HEIGHT = 520
DIRECTORY_NAME = "models/ov_artifacts"
BASE_MODEL_NAME = DIRECTORY_NAME + "/lraspp_mobilenet_v3_large"
weights_path = Path(BASE_MODEL_NAME + ".pth")



# Paths where ONNX and OpenVINO IR models will be stored.
onnx_path = weights_path.with_suffix(".onnx")
if not onnx_path.parent.exists():
    onnx_path.parent.mkdir()
ir_path = onnx_path.with_suffix(".xml")


'''
加载模型

通常, PyTorch 模型代表一个由状态字典初始化的 torch.nn.Module 类实例，获取预训练模型的典型步骤：


1. 创建模型类的实例
2. 加载检查点状态字典，其中包含预训练模型权重
3. 将模型转为评估模式，以切换某些操作到推理模式


该 torchvision 模块提供了一组用于模型类初始化的现成函数。
我们将使用 torchvision.models.segmentation.lraspp_mobilenet_v3_large 。
您可以直接使用权重枚举 LRASPP_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1 
将预训练模型权重传递给模型初始化函数。然而，出于演示目的，我们将单独创建它。
下载预训练权重并加载模型。如果您之前没有下载过模型，这可能需要一些时间。



'''


print(f'weights_path is\t\t {weights_path}')

print(f'url is\t\t\t {LRASPP_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1.url}')


if not weights_path.exists():
    print("Downloading the LRASPP MobileNetV3 model (if it has not been downloaded already)...")
    download_file(
        LRASPP_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1.url,
        filename=weights_path.name,
        directory=weights_path.parent,
    )




# 创建 model 对象
model = lraspp_mobilenet_v3_large()

# read state dict, use map_location argument to avoid a situation where weights are saved in cuda (which may not be unavailable on the system)
# 读取状态字典，并加载到 cpu 上
state_dict = torch.load(weights_path, map_location="cpu")


# load state dict to model
model.load_state_dict(state_dict)


# switch model from training to inference mode
model.eval()
print("Loaded PyTorch LRASPP MobileNetV3 model")



'''
ONNX 模型转换

将 PyTorch 模型转换为 ONNX

OpenVINO 支持以 ONNX 格式导出的 PyTorch 模型。
我们将使用 torch.onnx.export 函数获取 ONNX 模型，您可以在 PyTorch 文档中了解更多关于此功能的信息。
我们需要提供模型对象、模型跟踪的示例输入以及模型将保存的路径。

在提供示例输入时，不需要使用真实数据，具有指定形状的虚拟输入数据就足够了。
可选地，我们可以提供目标 onnx opset 以进行转换和/或根据文档中指定的其他参数（例如输入和输出名称或动态形状）。


有时会出现警告，但在大多数情况下是无害的，所以我们只需将其过滤掉即可。
当转换成功时，输出内容的最后一行将显示为： ONNX model exported to model/lraspp_mobilenet_v3_large.onnx.


'''



with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    if not onnx_path.exists():
        dummy_input = torch.randn(1, 3, IMAGE_HEIGHT, IMAGE_WIDTH)
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
        )
        print(f"ONNX model exported to {onnx_path}.")
    else:
        print(f"ONNX model\t {onnx_path} already exists.")


'''
将 ONNX 模型转换为 OpenVINO IR 格式


使用模型转换 API 将 ONNX 模型转换为 OpenVINO IR，
精度为 FP16 。模型保存在当前目录下。有关如何转换模型的更多信息，请参阅此页面。


'''



if not ir_path.exists():
    print("Exporting ONNX model to IR... This may take a few minutes.")
    ov_model = ov.convert_model(onnx_path)
    ov.save_model(ov_model, ir_path)
else:
    print(f"IR model\t {ir_path} already exists.")




'''
Show Results  显示结果


确认分割结果是否符合预期，通过比较 ONNX、OpenVINO IR 和 PyTorch 模型的预测结果。

加载并预处理输入图像
图像在通过网络之前需要进行归一化处理。



'''





def normalize(image: np.ndarray) -> np.ndarray:
    """
    Normalize the image to the given mean and standard deviation
    for CityScapes models.
    """
    image = image.astype(np.float32)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    image /= 255.0
    image -= mean
    image /= std
    return image




# Download the image from the openvino_notebooks storage
image_filename = Path("data/ov") / "coco.jpg"

if not image_filename.exists():
    download_file(
        "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/coco.jpg",
        directory="data",
    )



print("\n======================= 构建输入图片 \n")

image = cv2.cvtColor(cv2.imread(str(image_filename)), cv2.COLOR_BGR2RGB)


# plt.imshow(image)
# plt.show()


resized_image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))

# plt.imshow(resized_image)
# plt.show()

normalized_image = normalize(resized_image)

# plt.imshow(normalized_image)
# plt.show()




print("\n======================= 调整图片大小 \n")

print(resized_image.shape)







# Convert the resized images to network input shape.
input_image = np.expand_dims(np.transpose(resized_image, (2, 0, 1)), 0)


print("\n======================= 调整输入形状 \n")

print(input_image.shape)
normalized_input_image = np.expand_dims(np.transpose(normalized_image, (2, 0, 1)), 0)



'''
加载 OpenVINO IR 网络并在 ONNX 模型上运行推理


OpenVINO 运行时可以直接加载 ONNX 模型。首先，加载 ONNX 模型，进行推理并显示结果。
然后，使用 OpenVINO 转换器将模型转换为 OpenVINO 中间表示 (OpenVINO IR)，并在该模型上进行推理，然后在图像上显示结果。



1. OpenVINO 运行时中的 ONNX 模型


'''

print("\n======================= openvino 读取 ONNX 推理 \n")

# Instantiate OpenVINO Core
core = ov.Core()

# Read model to OpenVINO Runtime
model_onnx = core.read_model(model=onnx_path)



# Load model on device
compiled_model_onnx = core.compile_model(model=model_onnx, device_name='CPU')

# Run inference on the input image
res_onnx = compiled_model_onnx([normalized_input_image])[0]

print(res_onnx.shape)


'''
模型预测每个像素与特定标签对应的好坏的几率。为了得到每个像素的最高概率标签，
应应用 argmax 操作。之后，可以对每个标签应用颜色编码，以便更方便地可视化。

'''



voc_labels = [
    Label(index=0, color=(0, 0, 0), name="background"),
    Label(index=1, color=(128, 0, 0), name="aeroplane"),
    Label(index=2, color=(0, 128, 0), name="bicycle"),
    Label(index=3, color=(128, 128, 0), name="bird"),
    Label(index=4, color=(0, 0, 128), name="boat"),
    Label(index=5, color=(128, 0, 128), name="bottle"),
    Label(index=6, color=(0, 128, 128), name="bus"),
    Label(index=7, color=(128, 128, 128), name="car"),
    Label(index=8, color=(64, 0, 0), name="cat"),
    Label(index=9, color=(192, 0, 0), name="chair"),
    Label(index=10, color=(64, 128, 0), name="cow"),
    Label(index=11, color=(192, 128, 0), name="dining table"),
    Label(index=12, color=(64, 0, 128), name="dog"),
    Label(index=13, color=(192, 0, 128), name="horse"),
    Label(index=14, color=(64, 128, 128), name="motorbike"),
    Label(index=15, color=(192, 128, 128), name="person"),
    Label(index=16, color=(0, 64, 0), name="potted plant"),
    Label(index=17, color=(128, 64, 0), name="sheep"),
    Label(index=18, color=(0, 192, 0), name="sofa"),
    Label(index=19, color=(128, 192, 0), name="train"),
    Label(index=20, color=(0, 64, 128), name="tv monitor"),
]
VOCLabels = SegmentationMap(voc_labels)

# Convert the network result to a segmentation map and display the result.
result_mask_onnx = np.squeeze(np.argmax(res_onnx, axis=1)).astype(np.uint8)
viz_result_image(
    image,
    segmentation_map_to_image(result_mask_onnx, VOCLabels.get_colormap()),
    resize=True,
)


plt.show()



'''
2. OpenVINO 运行时中的 OpenVINO IR 模型

'''


print("\n======================= openvino 读取 openvino IR 推理 \n")

# Load the network in OpenVINO Runtime.
core = ov.Core()
model_ir = core.read_model(model=ir_path)
compiled_model_ir = core.compile_model(model=model_ir, device_name='CPU')

# Get input and output layers.
output_layer_ir = compiled_model_ir.output(0)

# Run inference on the input image.
res_ir = compiled_model_ir([normalized_input_image])[output_layer_ir]

result_mask_ir = np.squeeze(np.argmax(res_ir, axis=1)).astype(np.uint8)
viz_result_image(
    image,
    segmentation_map_to_image(result=result_mask_ir, colormap=VOCLabels.get_colormap()),
    resize=True,
)


plt.show()


import openvino.properties as props


devices = core.available_devices
for device in devices:
    device_name = core.get_property(device, props.device.full_name)
    print(f"{device}: {device_name}")


    