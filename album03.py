

'''
使用 Albumentations 增强目标检测任务中的边界框

Reference

https://albumentations.ai/docs/examples/example_bboxes/

Using Albumentations to augment bounding boxes for object detection tasks



'''



import random

import cv2
from matplotlib import pyplot as plt

import albumentations as A



import requests
import numpy as np
from PIL import Image


from PIL import ImageDraw 

from PIL import Image, ImageDraw, ImageFont
font = ImageFont.load_default()
font.size = 30







BOX_COLOR = (255, 0, 0) # Red
TEXT_COLOR = (255, 255, 255) # White








# image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(image_url, stream=True).raw)

image = Image.open('imgs/000000386298.jpg')

image_np = np.array(image)

print(image)



# 坐标框的坐标使用 coco 格式声明。每个坐标框用四个值 [x_min, y_min, width, height] 描述。

bboxes = [[5.66, 138.95, 147.09, 164.88], [366.7, 80.84, 132.8, 181.84]]
category_ids = [17, 18]

# We will use the mapping from category_id to the class name
# to visualize the class label for the bounding box on the image
category_id_to_name = {17: 'cat', 18: 'dog'}

# visualize(cv_image, bboxes, category_ids, category_id_to_name)



# draw = ImageDraw.Draw(image)




# for bbox, category_id in zip(bboxes, category_ids):
#     bbox = [int(i) for i in bbox]
#     print(bbox)
#     x = bbox[0]
#     y = bbox[1]
#     x2 = bbox[0]+bbox[2]
#     y2 = bbox[1]+bbox[3]
#     draw.rectangle((x, y, x2, y2 ), outline='blue', width=1)
#     draw.text((x+5,y+1), category_id_to_name[category_id], fill='blue', font= font )


# image.show()



'''
接收 coco 格式的 bbox [x, y, width, height]
绘制长方形时，转换为 pascal voc 格式，左上角坐标，右下角坐标
'''
def visualize(image: np.ndarray, bboxes, category_ids, category_id_to_name):

    augmented_image = Image.fromarray(image)
    draw = ImageDraw.Draw(augmented_image)

    for bbox, category_id in zip(bboxes, category_ids):
        bbox = [int(i) for i in bbox]
        print(bbox)
        x = bbox[0]
        y = bbox[1]
        x2 = bbox[0]+bbox[2]
        y2 = bbox[1]+bbox[3]
        draw.rectangle((x, y, x2, y2 ), outline='blue', width=1)
        draw.text((x+5,y+1), category_id_to_name[category_id], fill='blue', font= font )
    
    augmented_image.show()





visualize(image_np, bboxes, category_ids, category_id_to_name)



transform = A.Compose(
    [A.HorizontalFlip(p=0.5)],
    bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']),
)


random.seed(7)
transformed = transform(image=image_np, bboxes=bboxes, category_ids=category_ids)
visualize(
    transformed['image'],
    transformed['bboxes'],
    transformed['category_ids'],
    category_id_to_name,
)



# 再写一个例子

transform = A.Compose(
    [A.ShiftScaleRotate(p=0.5)],
    bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']),
)

random.seed(7)
transformed = transform(image=image_np, bboxes=bboxes, category_ids=category_ids)
visualize(
    transformed['image'],
    transformed['bboxes'],
    transformed['category_ids'],
    category_id_to_name,
)


# 定义复杂的增强


transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=0.3),
    ],
    bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']),
)


random.seed(7)
transformed = transform(image=image_np, bboxes=bboxes, category_ids=category_ids)
visualize(
    transformed['image'],
    transformed['bboxes'],
    transformed['category_ids'],
    category_id_to_name,
)



'''

min_area 和 min_visibility参数

如果变换后目标框小于 min_area 那么就放弃目标框
如果变换后目标框与原来目标框的比值小于 min_visibility 那么就放弃目标框




'''


transform = A.Compose(
    [A.CenterCrop(height=280, width=280, p=1)],
    bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']),
)



transformed = transform(image=image_np, bboxes=bboxes, category_ids=category_ids)
visualize(
    transformed['image'],
    transformed['bboxes'],
    transformed['category_ids'],
    category_id_to_name,
)


# min_area 最小面积设定为 4500 像素
transform = A.Compose(
    [A.CenterCrop(height=280, width=280, p=1)],
    bbox_params=A.BboxParams(format='coco', min_area=4500, label_fields=['category_ids']),
)


transformed = transform(image=image_np, bboxes=bboxes, category_ids=category_ids)
visualize(
    transformed['image'],
    transformed['bboxes'],
    transformed['category_ids'],
    category_id_to_name,
)



# 使用 min_visibility 定义增强管道

# 最后，我们将 min_visibility 设置为 0.3。
# 因此，如果输出边界框的面积小于原始面积的 30%，Albumentations 将不会返回该边界框。



transform = A.Compose(
    [A.CenterCrop(height=280, width=280, p=1)],
    bbox_params=A.BboxParams(format='coco', min_visibility=0.3, label_fields=['category_ids']),
)

transformed = transform(image=image_np, bboxes=bboxes, category_ids=category_ids)
visualize(
    transformed['image'],
    transformed['bboxes'],
    transformed['category_ids'],
    category_id_to_name,
)













