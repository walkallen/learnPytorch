

'''



目标检测是计算机视觉任务，用于检测图像中的实例（如人类、建筑物或汽车）。目标检测模型接收图像作为输入，
并输出检测到的对象的边界框和相关标签的坐标。一张图像可以包含多个对象，每个对象都有自己的边界框和标签
（例如，可以有一辆汽车和一栋建筑物），
每个对象可以出现在图像的不同部分（例如，图像可以有几辆汽车）。这项任务通常用于自动驾驶中检测行人、
路标和交通灯等物体。其他应用包括在图像中计数对象、图像搜索等。


在本指南中，您将学习如何：


微调结合卷积骨干和编码器-解码器 Transformer 的 DETR 模型，在 CPPE-5 数据集上进行。

使用您微调过的模型进行推理。



开始之前，我们将定义全局常量，即模型名称和图像大小。在本教程中，我们将使用条件 DETR 模型，因为它收敛速度更快。
请随意选择 transformers 库中可用的任何目标检测模型。


'''


MODEL_NAME = "microsoft/conditional-detr-resnet-50"  # or "facebook/detr-resnet-50"
IMAGE_SIZE = 480


'''
Load the CPPE-5 dataset  加载 CPPE-5 数据集

CPPE-5 数据集包含图像，其中标注了在 COVID-19 大流行期间医疗个人防护装备（PPE）。

开始加载数据集并从 train 创建 validation 分割

'''


from datasets import load_dataset

cppe5 = load_dataset("cppe-5")

print(cppe5)
print('\n')


if "validation" not in cppe5:
    split = cppe5["train"].train_test_split(0.15, seed=1337)
    cppe5["train"] = split["train"]
    cppe5["validation"] = split["test"]

print(cppe5)
print('\n')


'''
您将看到这个数据集包含 1000 张用于训练和验证集的图像以及一个包含 29 张图像的测试集。

了解数据，探索示例的形态。

'''

print(cppe5["train"][0])
print('\n')
'''

{
  'image_id': 366,
  'image': <PIL.PngImagePlugin.PngImageFile image mode=RGBA size=500x290>,
  'width': 500,
  'height': 500,
  'objects': {
    'id': [1932, 1933, 1934],
    'area': [27063, 34200, 32431],
    'bbox': [[29.0, 11.0, 97.0, 279.0],
      [201.0, 1.0, 120.0, 285.0],
      [382.0, 0.0, 113.0, 287.0]],
    'category': [0, 0, 0]
  }
}
    
'''


import numpy as np
import os
from PIL import Image, ImageDraw

image = cppe5["train"][2]["image"]
annotations = cppe5["train"][2]["objects"]
draw = ImageDraw.Draw(image)

categories = cppe5["train"].features["objects"].feature["category"].names

print("suhao categories is")
print(categories)
print('\n')

id2label = {index: x for index, x in enumerate(categories, start=0)}
label2id = {v: k for k, v in id2label.items()}

print("suhao id2label is")
print(id2label)
print('\n')

print("suhao label2id is")
print(label2id)
print('\n')

# 遍历识别到的物体 objects 中的每一个 id
for i in range(len(annotations["id"])):
    # box 是所有识别到的物体的其中一个
    box = annotations["bbox"][i]
    # class_idx 是对应的类别 id
    class_idx = annotations["category"][i]
    x, y, w, h = tuple(box)
    # Check if coordinates are normalized or not

    # 根据 box 中的具体数值是否正则化，做出不同处理
    if max(box) > 1.0:
        # Coordinates are un-normalized, no need to re-scale them
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)
    else:
        # Coordinates are normalized, re-scale them
        x1 = int(x * width)
        y1 = int(y * height)
        x2 = int((x + w) * width)
        y2 = int((y + h) * height)
    
    # 绘画正方形
    draw.rectangle((x, y, x + w, y + h), outline="red", width=1)

    # 绘画类别标签
    draw.text((x, y), id2label[class_idx], fill="red")

image.show()


'''
为了可视化与相关标签的边界框，您可以从数据集的元数据中获取标签，特别是 category 字段。
您还希望创建将标签 ID 映射到标签类别（ id2label ）以及相反映射的字典（ label2id ）。
您可以在设置模型时使用它们。包含这些映射将使您的模型在您将其分享到 Hugging Face Hub 时可由他人重用。
请注意，上述代码中绘制边界框的部分假设它处于 COCO 格式 (x_min, y_min, width, height) 。
它必须调整以适应其他格式如 (x_min, y_min, x_max, y_max) 。

作为熟悉数据的最后一步，探索潜在问题。对象检测数据集的一个常见问题是边界框“拉伸”到图像边缘之外。
这种“失控”的边界框可能在训练期间引发错误，应该予以解决。在这个数据集中有几个存在此问题的示例。
为了使本指南内容简单，我们将在以下转换中将 clip=True 设置为 BboxParams 。




预处理数据

为了微调模型，您必须预处理您打算使用的数据，以确保与预训练模型使用的精确方法相匹配。
AutoImageProcessor 负责处理图像数据以创建 pixel_values 、 pixel_mask 和 labels ,
这些是 DETR 模型可以训练的。图像处理器有一些属性，您不必担心：


image_mean = [0.485, 0.456, 0.406 ]
image_std = [0.229, 0.224, 0.225]

这些是用于模型预训练期间图像归一化的均值和标准差。这些值在执行推理或微调预训练图像模型时至关重要。


'''