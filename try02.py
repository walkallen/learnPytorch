from datasets import DatasetDict, Dataset
from pycocotools.coco import COCO
import os
from PIL import Image

def load_coco_dataset(image_dir, annotation_path):
    # 加载 COCO 标注文件
    coco = COCO(annotation_path)

    # 获取所有图片 ID
    image_ids = coco.getImgIds()

    # 准备数据
    data = []
    for img_id in image_ids:
        # 获取图片信息
        img_info = coco.loadImgs(img_id)[0]
        file_name = img_info['file_name']
        image_path = os.path.join(image_dir, file_name)

        # 获取标注信息
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        # 提取标注（边界框、类别等）
        bboxes = []
        categories = []
        for ann in anns:
            bboxes.append(ann['bbox'])  # COCO 格式: [x_min, y_min, width, height]
            categories.append(ann['category_id'])

        # 添加到数据列表
        data.append({
            'image': Image.open(image_path).convert('RGB'),  # 加载图片
            'image_id': img_id,
            'file_name': file_name,
            'width': img_info['width'],
            'height': img_info['height'],
            'objects': {  # 标注信息
                'bbox': bboxes,
                'categories': categories,
            }
        })

    # 创建 Hugging Face Dataset
    return Dataset.from_list(data)

# 加载训练集和验证集
image_dir_train = 'path/to/coco/train2017'
annotation_path_train = 'path/to/coco/annotations/instances_train2017.json'
train_dataset = load_coco_dataset(image_dir_train, annotation_path_train)

image_dir_val = 'path/to/coco/val2017'
annotation_path_val = 'path/to/coco/annotations/instances_val2017.json'
val_dataset = load_coco_dataset(image_dir_val, annotation_path_val)

# 创建 DatasetDict
coco_dataset = DatasetDict({
    'train': train_dataset,
    'validation': val_dataset,
})

# 查看数据集
print(coco_dataset)