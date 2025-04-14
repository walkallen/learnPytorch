

'''
Reference 

https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/hello-world/hello-world.ipynb




Hello Image Classification
图像分类简介


本教程中使用了来自 Open Model Zoo 的预训练 MobileNetV3 模型。有关如何创建 OpenVINO IR 模型的更多信息，请参阅 TensorFlow 到 OpenVINO 教程。



Table of contents:  目录：
Imports  导入
Download the Model and data samples
下载模型和数据样本
Select inference device  选择推理设备
Load the Model  加载模型
Load an Image  加载图片
Do Inference  进行推理



'''

from pathlib import Path


import cv2
import matplotlib.pyplot as plt
import numpy as np
import openvino as ov




from ov_notebook_utils import download_file, device_widget




from ov_notebook_utils import collect_telemetry


base_artifacts_dir = Path("./models/ov_artifacts").expanduser()

print(base_artifacts_dir)

model_name = "v3-small_224_1.0_float"
model_xml_name = f"{model_name}.xml"
model_bin_name = f"{model_name}.bin"




model_xml_path = base_artifacts_dir / model_xml_name

print(model_xml_path)

base_url = "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/models/mobelinet-v3-tf/FP32/"



if not model_xml_path.exists():
    download_file(base_url + model_xml_name, model_xml_name, base_artifacts_dir)
    download_file(base_url + model_bin_name, model_bin_name, base_artifacts_dir)
else:
    print(f"{model_name} already downloaded to {base_artifacts_dir}")




core = ov.Core()
supported_devices = core.available_devices

print(supported_devices)

# 读取 openvino 模型
model = core.read_model(model=model_xml_path)


compiled_model = core.compile_model(model=model, device_name='CPU')

output_layer = compiled_model.output(0)

print(output_layer)



image_filename = Path("data/ov/coco.jpg")


if not image_filename.exists():
    download_file(
        "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/coco.jpg",
        directory="data/ov",
    )


# The MobileNet model expects images in RGB format.
image = cv2.cvtColor(cv2.imread(filename=str(image_filename)), code=cv2.COLOR_BGR2RGB)

# print(image)

# Resize to MobileNet image shape.
input_image = cv2.resize(src=image, dsize=(224, 224))



# Reshape to model input shape.
input_image = np.expand_dims(input_image, 0)
plt.imshow(image)
# plt.show()





# Do Inference  进行推理


imagenet_filename = Path("data/ov/imagenet_2012.txt")

if not imagenet_filename.exists():
    download_file(
        "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/datasets/imagenet/imagenet_2012.txt",
        directory="data/ov",
    )


result_infer = compiled_model([input_image])[output_layer]

print(result_infer)
result_index = np.argmax(result_infer)
print(f'argmax index is {result_index}')


imagenet_classes = imagenet_filename.read_text().splitlines()

# The model description states that for this model, class 0 is a background.
# Therefore, a background must be added at the beginning of imagenet_classes.
imagenet_classes = ["background"] + imagenet_classes

print(
imagenet_classes[result_index]
)

# n02099267 flat-coated retriever 








