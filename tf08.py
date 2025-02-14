

import torch
from torch.utils.data import DataLoader
from typing import Any, List, Mapping, Tuple, Union

from torchmetrics.detection.mean_ap import MeanAveragePrecision

import transformers
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForObjectDetection,
    SchedulerType,
    get_scheduler,
)



def evaluation_loop(
    model: torch.nn.Module,                 # 类型提示， model 参数期望接收一个 PyTorch 神经网络模型 (torch.nn.Module 的实例)。
    image_processor: AutoImageProcessor,    # AutoImageProcessor 负责将原始图像数据转换为模型可以接受的输入格式，例如调整图像大小、归一化像素值等
    accelerator: Accelerator,               # Accelerator 可以自动处理设备放置、分布式训练同步等细节
    dataloader: DataLoader,                 # DataLoader 用于批量加载评估数据集， 提供迭代访问评估数据批次的功能
    id2label: Mapping[int, str],            # id2label 参数期望接收一个映射 (例如字典)， 将类别 ID (整数) 映射到类别名称 (字符串)。 这用于将模型预测的类别 ID 转换为可读的类别名称，并在评估结果中展示
) -> dict:                                  # 返回类型提示， 表明该函数预计返回一个 Python 字典
    
    model.eval()    # 评估模式
    metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True)        # 这行代码初始化了一个 Mean Average Precision (mAP) 的评估指标计算器
                                                                                # "xyxy" 格式通常表示边界框的坐标为 [x_min, y_min, x_max, y_max]， 即左上角和右下角坐标
                                                                                # 参数 class_metrics=True 表示除了计算总体的 mAP 值之外， 还要计算每个类别的 AP 值


    for step, batch in enumerate(tqdm(dataloader, disable=not accelerator.is_local_main_process)):
        with torch.no_grad():
            outputs = model(**batch)

        # For metric computation we need to collect ground truth and predicted boxes in the same format

        # 1. Collect predicted boxes, classes, scores
        # image_processor convert boxes from YOLO format to Pascal VOC format
        # ([x_min, y_min, x_max, y_max] in absolute coordinates)
        image_size = torch.stack([example["orig_size"] for example in batch["labels"]], dim=0)
        predictions = image_processor.post_process_object_detection(outputs, threshold=0.0, target_sizes=image_size)
        predictions = nested_to_cpu(predictions)

        # 2. Collect ground truth boxes in the same format for metric computation
        # Do the same, convert YOLO boxes to Pascal VOC format
        target = []
        for label in batch["labels"]:
            label = nested_to_cpu(label)
            boxes = convert_bbox_yolo_to_pascal(label["boxes"], label["orig_size"])
            labels = label["class_labels"]
            target.append({"boxes": boxes, "labels": labels})

        metric.update(predictions, target)

    metrics = metric.compute()

    # Replace list of per class metrics with separate metric for each class
    classes = metrics.pop("classes")
    map_per_class = metrics.pop("map_per_class")
    mar_100_per_class = metrics.pop("mar_100_per_class")
    for class_id, class_map, class_mar in zip(classes, map_per_class, mar_100_per_class):
        class_name = id2label[class_id.item()]
        metrics[f"map_{class_name}"] = class_map
        metrics[f"mar_100_{class_name}"] = class_mar

    # Convert metrics to float
    metrics = {k: round(v.item(), 4) for k, v in metrics.items()}

    return metrics



