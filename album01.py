

import albumentations as A
import requests
import numpy as np
from PIL import Image



image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(image_url, stream=True).raw)

image.show()

image_np = np.array(image)

print(image)

# 读取图像
# image = cv2.imread('image.jpg')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 定义增强管道
transform = A.Compose([
    A.RandomCrop(width=256, height=256),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
])

# 应用增强, transform 接收 numpy 数组
augmented = transform(image=image_np)
print("增强以后，返回一个词典，但里面只有一个 image 键值对，对应一个 numpy 增强结果图片")
print(augmented)
augmented_image_np = augmented['image']

augmented_image = Image.fromarray(augmented_image_np)


print(augmented_image)

augmented_image.show()
