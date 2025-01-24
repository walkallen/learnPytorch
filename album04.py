'''
Reference

https://albumentations.ai/docs/examples/example_bboxes2/


How to use Albumentations for detection tasks if you need to keep all bounding boxes
如何使用 Albumentations 进行检测任务，如果您需要保留所有边界框


某些增强如 RandomCrop 和 CenterCrop 可能会将图像转换，使其不包含所有原始边界框。
本例展示了如何使用名为 RandomSizedBBoxSafeCrop 的转换来裁剪图像的一部分，但保留原始图像中的所有边界框。



'''



import random

import cv2
from matplotlib import pyplot as plt

import albumentations as A

import numpy as np

from PIL import Image, ImageDraw, ImageFont
font = ImageFont.load_default()
font.size = 30


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



image = Image.open('imgs/000000386298.jpg')

image = np.array(image)

print(image)



bboxes = [[5.66, 138.95, 147.09, 164.88], [366.7, 80.84, 132.8, 181.84]]
category_ids = [17, 18]

# We will use the mapping from category_id to the class name
# to visualize the class label for the bounding box on the image
category_id_to_name = {17: 'cat', 18: 'dog'}



visualize(image, bboxes, category_ids, category_id_to_name)


'''
使用 RandomSizedBBoxSafeCrop 保留原始图像中所有边界框


RandomSizedBBoxSafeCrop 裁剪图像的随机部分。它确保裁剪的部分将包含原始图像中的所有边界框。
然后，变换将裁剪部分调整到相应参数指定的宽度和高度。 erosion_rate 参数控制裁剪后原始边界框
可能丢失的面积。 erosion_rate = 0.2 表示增强后的边界框面积可能比原始边界框面积小 20%。



'''

transform = A.Compose(
    [A.RandomSizedBBoxSafeCrop(width=448, height=336, erosion_rate=0.2)],
    bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']),
)

random.seed(7)
transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)
visualize(
    transformed['image'],
    transformed['bboxes'],
    transformed['category_ids'],
    category_id_to_name,
)


'''

用边界框增强输入图像

我们将随机种子固定以进行可视化目的，因此增强始终会产生相同的结果。在真实的计算机视觉管道中，
您不应该在将转换应用到图像之前固定随机种子，因为在这种情况下，管道将始终输出相同的图像。图像增强的目的是每次使用不同的转换。

更多使用不同随机种子的示例

'''



random.seed(3)
transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)
visualize(
    transformed['image'],
    transformed['bboxes'],
    transformed['category_ids'],
    category_id_to_name,
)









