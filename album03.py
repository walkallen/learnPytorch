

'''
使用 Albumentations 增强目标检测任务中的边界框


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


def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):
    """Visualizes a single bounding box on the image"""
    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35,
        color=TEXT_COLOR,
        lineType=cv2.LINE_AA,
    )
    return img


def visualize(image, bboxes, category_ids, category_id_to_name):
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name)
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imshow(img)

def visualize_pil(image, bboxes, category_ids, category_id_to_name):

    draw = ImageDraw.Draw(image)


    # img = image.copy()

    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name)
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imshow(img)



image_url = "http://images.cocodataset.org/val2017/000000386298.jpg"
image = Image.open(requests.get(image_url, stream=True).raw)

image_np = np.array(image)

print(image)

cv_image = image_np
cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
# cv2.imshow('opencv', cv_image)
# cv2.waitKey(0)

# 坐标框的坐标使用 coco 格式声明。每个坐标框用四个值 [x_min, y_min, width, height] 描述。

bboxes = [[5.66, 138.95, 147.09, 164.88], [366.7, 80.84, 132.8, 181.84]]
category_ids = [17, 18]

# We will use the mapping from category_id to the class name
# to visualize the class label for the bounding box on the image
category_id_to_name = {17: 'cat', 18: 'dog'}

# visualize(cv_image, bboxes, category_ids, category_id_to_name)



draw = ImageDraw.Draw(image)




for bbox, category_id in zip(bboxes, category_ids):
    bbox = [int(i) for i in bbox]
    draw.rectangle((bbox[0], bbox[1], bbox[0]+bbox[2], bbox[0]+bbox[3] ), outline='blue', width=1)



image.show()















