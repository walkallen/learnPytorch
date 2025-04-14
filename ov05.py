
'''
OpenVINO™ 运行时 API 教程

本笔记本解释了 OpenVINO 运行时 API 的基础知识。

笔记本分为几个部分，每个部分都有标题。下一个单元包含安装和导入的全局要求。
每个部分都是独立的，不依赖于任何前面的部分。本教程中使用的所有模型都作为示例提供。
这些模型文件可以被您自己的模型文件替换。具体的输出结果将不同，但过程是相同的。


目录

加载 OpenVINO 运行时并显示信息
加载模型
    OpenVINO IR 模型
    ONNX 模型
    PaddlePaddle 模型
    TensorFlow 模型
    TensorFlow Lite 模型
    PyTorch 模型

获取模型信息
    模型输入
    模型输出

在模型上进行推理
重新塑形和调整大小
    修改图像大小
    修改批大小

缓存模型

'''


import requests
from pathlib import Path

from ov_notebook_utils import download_file, load_image, device_widget

import openvino as ov

'''
OpenVINO 运行时可以在设备上加载网络。在此上下文中，
设备指的是 CPU、英特尔 GPU、Neural Compute Stick 2 等。 

available_devices 属性显示了您系统中的可用设备。"FULL_DEVICE_NAME" 
选项显示 core.get_property() 的设备名称。


'''

core = ov.Core()




import openvino.properties as props


devices = core.available_devices

for device in devices:
    device_name = core.get_property(device, props.device.full_name)
    print(f"{device}: {device_name}")


'''
加载模型

初始化 OpenVINO Runtime 后，首先使用 read_model() 读取模型文件，
然后使用 compile_model() 方法将其编译到指定的设备上。


OpenVINO™ 支持多种模型格式，并允许开发者使用专门为此任务设计的工具将它们转换为 OpenVINO IR 格式。


OpenVINO IR Model 

一个 OpenVINO IR（中间表示）模型由一个包含网络拓扑信息的 .xml 文件
和一个包含权重和偏置二进制数据的 .bin 文件组成。

OpenVINO IR 格式的模型是通过使用模型转换 API 获得的。 
read_model() 函数期望 .bin 权重文件与 .xml 文件具有相同的文件名，并且位于同一目录中： 
model_weights_file == Path(model_xml).with_suffix(".bin") 。

如果情况如此，指定权重文件是可选的。如果权重文件具有不同的文件名，
则可以使用 read_model() 中的 weights 参数进行指定。



OpenVINO 模型转换 API 工具用于将模型转换为 OpenVINO IR 格式。
模型转换 API 读取原始模型并创建 OpenVINO IR 模型（ .xml 和 .bin 文件），
以便在没有格式转换延迟的情况下进行推理。可选地，模型转换 API 可以调整模型以使其更适合推理，
例如，通过交替输入形状、嵌入预处理和截断训练部分。

有关如何使用模型转换 API 将现有的 TensorFlow、PyTorch 或 ONNX 模型
转换为 OpenVINO IR 格式的信息，请参阅 tensorflow-to-openvino 和 pytorch-onnx-to-openvino 笔记本。



'''



ir_model_url = "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/models/002-example-models/"
ir_model_name_xml = "classification.xml"
ir_model_name_bin = "classification.bin"

if not Path("./models/ov_artifacts/" + ir_model_name_xml).exists():
    download_file(ir_model_url + ir_model_name_xml, filename=ir_model_name_xml, directory="./models/ov_artifacts/")
    download_file(ir_model_url + ir_model_name_bin, filename=ir_model_name_bin, directory="./models/ov_artifacts/")




import openvino as ov

core = ov.Core()
classification_model_xml = "./models/ov_artifacts/classification.xml"

model = core.read_model(model=classification_model_xml)
compiled_model = core.compile_model(model=model, device_name=device.value)


















