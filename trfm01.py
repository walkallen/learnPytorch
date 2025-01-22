

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
from datasets import load_from_disk


# cppe5 = load_dataset("cppe-5")

# cppe5.save_to_disk('/mnt/SD1T/suhao/learnPytorchbak/learnPytorch/data_cppe5')

cppe5 = load_from_disk("/mnt/SD1T/suhao/learnPytorchbak/learnPytorch/data_cppe5")

print(cppe5)
print('\n')


if "validation" not in cppe5:
    split = cppe5["train"].train_test_split(0.15, seed=1337)
    cppe5["train"] = split["train"]
    cppe5["validation"] = split["test"]

print(cppe5)
print('\n')


'''

DatasetDict({
    train: Dataset({
        features: ['image_id', 'image', 'width', 'height', 'objects'],
        num_rows: 850
    })
    test: Dataset({
        features: ['image_id', 'image', 'width', 'height', 'objects'],
        num_rows: 29
    })
    validation: Dataset({
        features: ['image_id', 'image', 'width', 'height', 'objects'],
        num_rows: 150
    })
})

您将看到这个数据集包含 1000 张用于训练和验证集的图像以及一个包含 29 张图像的测试集。

了解数据，探索示例的形态。

'''
print('显示 cppe 里的一个数据')
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
    'bbox': 
     [[29.0, 11.0, 97.0, 279.0],
      [201.0, 1.0, 120.0, 285.0],
      [382.0, 0.0, 113.0, 287.0]],
    'category': [0, 0, 0]
  }
}

您可能会注意到 bbox 字段遵循 COCO 格式，这是 DETR 模型期望的格式。然而， 
objects 内部字段的分组与 DETR 所需的注释格式不同。在使用这些数据进行训练之前，您需要应用一些预处理转换。


    
'''


import numpy as np
import os
from PIL import Image, ImageDraw

image = cppe5["train"][2]["image"]
annotations = cppe5["train"][2]["objects"]
draw = ImageDraw.Draw(image)

print(cppe5["train"].features)
print("\n")
'''
{
    'image_id': Value(dtype='int64', id=None), 
    'image': Image(mode=None, decode=True, id=None), 
    'width': Value(dtype='int32', id=None), 
    'height': Value(dtype='int32', id=None), 
    'objects': Sequence(feature={'id': Value(dtype='int64', id=None), 
    'area': Value(dtype='int64', id=None), 
    'bbox': Sequence(feature=Value(dtype='float32', id=None), length=4, id=None), 
    'category': ClassLabel(names=['Coverall', 'Face_Shield', 'Gloves', 'Goggles', 'Mask'], id=None)}, 
    length=-1, id=None)
}

通过 features 中的 category 获得标签
'''

print( cppe5["train"].features["objects"]  )
print('\n')

'''
Sequence(
    feature={'id': Value(dtype='int64', id=None), 
    'area': Value(dtype='int64', id=None), 
    'bbox': Sequence(feature=Value(dtype='float32', id=None), length=4, id=None), 
    'category': ClassLabel(names=['Coverall', 'Face_Shield', 'Gloves', 'Goggles', 'Mask'], id=None)}, 
    length=-1, id=None)

'''


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

从与您要微调的模型相同的检查点实例化图像处理器。
'''


from transformers import AutoImageProcessor

MAX_SIZE = IMAGE_SIZE

image_processor = AutoImageProcessor.from_pretrained(
    MODEL_NAME,
    do_resize=True,
    size={"max_height": MAX_SIZE, "max_width": MAX_SIZE},
    do_pad=True,
    pad_size={"height": MAX_SIZE, "width": MAX_SIZE},
)



'''
在将图像传递给 image_processor 之前，对数据集应用两种预处理转换：

Augmenting images  增强图像

Reformatting annotations to meet DETR expectations  重整标注以满足 DETR 期望



首先，为了确保模型不会在训练数据上过拟合，你可以使用任何数据增强库来应用图像增强。
这里我们使用 Albumentations。这个库确保变换会影响图像并相应地更新边界框。
🤗 Datasets 库文档有关于如何为对象检测增强图像的详细指南，并使用相同的示例数据集。
对图像应用一些几何和颜色变换。要探索更多增强选项，请查看 Albumentations 演示空间。




'''


import albumentations as A

train_augment_and_transform = A.Compose(
    [
        A.Perspective(p=0.1),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.1),
    ],
    bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True, min_area=25),
)

validation_transform = A.Compose(
    [A.NoOp()],
    bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True),
)



'''

该 image_processor 期望注释采用以下格式： 
{
    'image_id': int, 
    'annotations': List[Dict]
} 
其中每个字典都是一个 COCO 对象注释。让我们添加一个函数来重新格式化单个示例的注释：

'''

def format_image_annotations_as_coco(image_id, categories, areas, bboxes):
    """Format one set of image annotations to the COCO format

    Args:
        image_id (str): image id. e.g. "0001"
        categories (List[int]): list of categories/class labels corresponding to provided bounding boxes
        areas (List[float]): list of corresponding areas to provided bounding boxes
        bboxes (List[Tuple[float]]): list of bounding boxes provided in COCO format
            ([center_x, center_y, width, height] in absolute coordinates)

    Returns:
        dict: {
            "image_id": image id,
            "annotations": list of formatted annotations
        }
    """
    annotations = []
    for category, area, bbox in zip(categories, areas, bboxes):
        formatted_annotation = {
            "image_id": image_id,
            "category_id": category,
            "iscrowd": 0,
            "area": area,
            "bbox": list(bbox),
        }
        annotations.append(formatted_annotation)

    return {
        "image_id": image_id,
        "annotations": annotations,
    }



'''
现在您可以将图像和注释变换组合起来，用于批量示例：

'''

def augment_and_transform_batch(examples, transform, image_processor, return_pixel_mask=False):
    """Apply augmentations and format annotations in COCO format for object detection task"""

    images = []
    annotations = []
    for image_id, image, objects in zip(examples["image_id"], examples["image"], examples["objects"]):
        image = np.array(image.convert("RGB"))

        # apply augmentations
        output = transform(image=image, bboxes=objects["bbox"], category=objects["category"])
        images.append(output["image"])

        # format annotations in COCO format
        formatted_annotations = format_image_annotations_as_coco(
            image_id, output["category"], objects["area"], output["bboxes"]
        )
        annotations.append(formatted_annotations)

    # Apply the image processor transformations: resizing, rescaling, normalization
    result = image_processor(images=images, annotations=annotations, return_tensors="pt")

    if not return_pixel_mask:
        result.pop("pixel_mask", None)

    return result


'''

将此预处理函数应用于整个数据集，使用 🤗 Datasets 的 with_transform 方法。此方法在加载数据集的元素时即时应用转换。

在此阶段，您可以查看数据集中的一个示例在转换后的样子。您应该看到一个带有 pixel_values 的张量，一个带有 pixel_mask 的张量，以及一个带有 labels 的张量。
'''


from functools import partial

# Make transform functions for batch and apply for dataset splits
train_transform_batch = partial(
    augment_and_transform_batch, transform=train_augment_and_transform, image_processor=image_processor
)
validation_transform_batch = partial(
    augment_and_transform_batch, transform=validation_transform, image_processor=image_processor
)

cppe5["train"] = cppe5["train"].with_transform(train_transform_batch)
cppe5["validation"] = cppe5["validation"].with_transform(validation_transform_batch)
cppe5["test"] = cppe5["test"].with_transform(validation_transform_batch)

print(
cppe5["train"][15]
)






'''
您已成功增强单个图像并准备它们的注释。然而，预处理尚未完成。
在最后一步，创建一个自定义 collate_fn 以批量组合图像。
将图像（现在是 pixel_values ）填充到批量中最大的图像大小，
并创建相应的 pixel_mask 以指示哪些像素是真实的(1)以及哪些是填充(0)。

'''


import torch

def collate_fn(batch):
    data = {}
    data["pixel_values"] = torch.stack([x["pixel_values"] for x in batch])
    data["labels"] = [x["labels"] for x in batch]
    if "pixel_mask" in batch[0]:
        data["pixel_mask"] = torch.stack([x["pixel_mask"] for x in batch])
    return data



'''
准备计算 mAP 的函数

目标检测模型通常使用一组 COCO 风格的指标进行评估。
我们将使用 torchmetrics 来计算 mAP （平均精度）和 mAR （平均召回率）指标，
并将其封装到 compute_metrics 函数中，以便在训练器中进行评估。

中间用于训练的框的格式是 YOLO （归一化），但我们将计算 Pascal VOC （绝对）
格式中框的度量，以正确处理框面积。让我们定义一个将边界框转换为 Pascal VOC 格式的函数：


'''


from transformers.image_transforms import center_to_corners_format

def convert_bbox_yolo_to_pascal(boxes, image_size):
    """
    Convert bounding boxes from YOLO format (x_center, y_center, width, height) in range [0, 1]
    to Pascal VOC format (x_min, y_min, x_max, y_max) in absolute coordinates.

    Args:
        boxes (torch.Tensor): Bounding boxes in YOLO format
        image_size (Tuple[int, int]): Image size in format (height, width)

    Returns:
        torch.Tensor: Bounding boxes in Pascal VOC format (x_min, y_min, x_max, y_max)
    """
    # convert center to corners format
    boxes = center_to_corners_format(boxes)

    # convert to absolute coordinates
    height, width = image_size
    boxes = boxes * torch.tensor([[width, height, width, height]])

    return boxes


