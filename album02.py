

import albumentations as A
import requests
import numpy as np
from PIL import Image


'''
边界框是标记图像上对象的矩形。边界框注释有多个格式。每个格式都使用其特定的边界框坐标表示。
Albumentations 支持四种格式： pascal_voc ， albumentations ， coco ，和 yolo 。


作为一个例子，我们将使用来自名为“场景中的常见物体”数据集的一张图片。
它包含一个标记猫的边界框。图片宽度为 640 像素，高度为 480 像素。
边界框的宽度为 322 像素，高度为 117 像素。


边界框的四个角坐标如下：左上角是 (x, y) 或 (x_min, y_min) ，
右上角是 (98px, 345px) 或 (x_max, y_min) ，左下角是 (420px, 345px) 或 (x_min, y_max) ，
右下角是 (98px, 462px) 或 (x_max, y_max) 。如您所见，边界框四个角的坐标是以图像左上角为基准计算的，
该点的坐标为 (420px, 462px) 或 (x, y) 。



'''


image_url = "https://albumentations.ai/docs/images/getting_started/augmenting_bboxes/bbox_example.jpg"
image = Image.open(requests.get(image_url, stream=True).raw)

image_np = np.array(image)


image.show()


'''
pascal_voc 是 Pascal VOC 数据集使用的格式。
边界框的坐标用四个像素值进行编码： [x_min, y_min, x_max, y_max] 。 
x_min 和 y_min 是边界框左上角的坐标。 x_max 和 y_max 是边界框右下角的坐标。

坐标示例边界框的格式为 [98, 345, 420, 462] 。


albumentations 与 pascal_voc 类似，因为它也使用四个值 [x_min, y_min, x_max, y_max] 
来表示边界框。但与 pascal_voc 不同， albumentations 使用归一化值。
为了归一化值，我们将像素坐标的 x 轴和 y 轴坐标除以图像的宽度和高度。


坐标示例边界框的格式为 [98 / 640, 345 / 480, 420 / 640, 462 / 480] ，
它们是 [0.153125, 0.71875, 0.65625, 0.9625] 。







coco 是 Common Objects in Context COCO 数据集所使用的格式。
在 coco 中，边界框由像素 [x_min, y_min, width, height] 中的四个值定义。
它们是边界框左上角的坐标以及边界框的宽度和高度。

坐标示例边界框的格式为 [98, 345, 322, 117] 。






在 yolo 中，边界框由四个值 [x_center, y_center, width, height] 表示。 
x_center 和 y_center 是边界框中心的归一化坐标。为了使坐标归一化，我们取 x 和 y 的像素值，
这些值标记了边界框在 x 轴和 y 轴上的中心。然后我们将 x 的值除以图像的宽度，将 y 的值除以图像的高度。 
width 和 height 表示边界框的宽度和高度。它们也进行了归一化。


坐标示例边界框的格式为 [((420 + 98) / 2) / 640, ((462 + 345) / 2) / 480, 322 / 640, 117 / 480] ，
它们是 [0.4046875, 0.840625, 0.503125, 0.24375] 。




总结

pascal_voc      [X_min, Y_min, X_max, Y_max]
albumentations  normalized[X_min, Y_min, X_max, Y_max]
coco            [X_min, Y_min, width, height]
yolo            normalized[x_center, y_center, width, height]



边界框增强

与图像和掩码增强一样，边界框增强的过程包括 4 个步骤。

1. 您导入所需的库。
2. 您定义了一个增强管道。
3. 您从磁盘读取图像和边界框。
4. 您将图像和边界框传递给增强管道，并接收增强后的图像和框。



'''



import albumentations as A
import cv2




transform = A.Compose(
    [   A.RandomCrop(width=450, height=450),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
    ], bbox_params=A.BboxParams(format='coco')
)



'''

请注意，与图像和掩码增强不同， Compose 现在有一个额外的参数 bbox_params 。
您需要传递一个 A.BboxParams 的实例给该参数。 A.BboxParams 指定与边界框一起工作的设置。 
format 设置边界框坐标的格式。

它可以是 pascal_voc 、 albumentations 、 coco 或 yolo 。
此值是必需的，因为 Albumentation 需要知道边界框坐标的源格式，以便正确应用增强。

除了 format ， A.BboxParams 还支持一些其他设置。



'''

transform = A.Compose(
    [
        A.RandomCrop(width=450, height=450),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
    ], 
    bbox_params=A.BboxParams(format='coco', 
                                min_area=1024, 
                                min_visibility=0.1, 
                                label_fields=['class_labels'])
)


'''
min_area 和 min_visibility 参数控制在增强后边界框大小发生变化时，
Albumentations 应对增强边界框执行的操作。边界框的大小可能会发生变化，
如果您应用空间增强，例如裁剪图像的一部分或调整图像大小。

min_area 是像素值。如果增强后的边界框面积小于 min_area , 
Albumentations 将丢弃该框。因此，返回的增强边界框列表将不包含该边界框。

min_visibility 是一个介于 0 和 1 之间的值。如果增强后的边界框面积与 
原来边界框的比率小于 min_visibility , Albumentations 将丢弃该框。
因此，如果增强过程切掉了边界框的大部分，那么该框将不会出现在增强边界框的返回列表中。



除了坐标外，每个边界框都应该有一个关联的类别标签，以说明哪个对象位于边界框内。为边界框传递标签有两种方式。

假设你有一个包含三个对象的示例图像： dog 、 cat 和 sports ball 。这些对象的边界框坐标以 coco 格式
分别为 [23, 74, 295, 388] 、 [377, 294, 252, 161] 和 [333, 421, 49, 49] 。


1. 您可以通过将标签作为附加值添加到坐标列表中来传递边界框坐标。

对于上面的图像，带有类别标签的边界框将变为 [23, 74, 295, 388, 'dog'] 、 
[377, 294, 252, 161, 'cat'] 和 [333, 421, 49, 49, 'sports ball'] 。

2. 您可以将边界框的标签作为单独的列表传递（这是首选方式）。

例如，如果您有三个边界框如 [23, 74, 295, 388] 、 [377, 294, 252, 161] 和 [333, 421, 49, 49] ，
您可以创建一个包含类似 ['cat', 'dog', 'sports ball'] 或 [18, 17, 37] 值的单独列表，
这些值包含那些边界框的类别标签。接下来，将带有类别标签的列表作为单独的参数传递给 transform 函数。
Albumentations 需要知道所有这些带有类别标签的列表的名称，以便正确地将它们与增强后的边界框连接起来。
然后，如果由于不再可见而删除了边界框，Albumentations 也将删除该框的类别标签。使用 label_fields 参数
设置 transform 中所有将包含边界框标签描述的参数的名称（更多内容请参阅第 4 步）。


边界框可以以不同的序列化格式存储在磁盘上：JSON、XML、YAML、CSV 等。因此，读取边界框的代码取决于磁盘上数据的实际格式。
读取磁盘中的数据后，您需要为 Albumentations 准备边界框。

读取磁盘中的数据后，您需要为 Albumentations 准备边界框。


Albumentations 期望边界框以列表的列表形式表示。每个列表包含关于单个边界框的信息。
边界框定义应至少包含四个元素，代表该边界框的坐标。这四个值的实际含义取决于边界框的格式
（ pascal_voc 、 albumentations 、 coco 或 yolo ）。除了四个坐标外，每个边界框的
定义可能还包含一个或多个额外值。可以使用这些额外值来存储有关边界框的附加信息，
例如框内对象的类别标签。在增强过程中，Albumentations 不会处理这些额外值。
库将按原样返回它们，以及增强后边界框的更新坐标。


第 4 步。将图像和边界框传递给增强管道，并接收增强后的图像和框。

'''


bboxes = [
    [23, 74, 295, 388],
    [377, 294, 252, 161],
    [333, 421, 49, 49],
]



bboxes = [
    [23, 74, 295, 388, 'dog'],
    [377, 294, 252, 161, 'cat'],
    [333, 421, 49, 49, 'sports ball'],
]



bboxes = [
    [23, 74, 295, 388, 'dog', 'animal'],
    [377, 294, 252, 161, 'cat', 'animal'],
    [333, 421, 49, 49, 'sports ball', 'item'],
]


'''
接下来，您将图像及其边界框传递给 transform 函数，并接收增强后的图像和边界框。
'''


transformed = transform(image=image, bboxes=bboxes)
transformed_image = transformed['image']
transformed_bboxes = transformed['bboxes']



'''

2. 将类别标签通过单独的参数传递给 transform （推荐方式）。

'''


bboxes = [
    [23, 74, 295, 388],
    [377, 294, 252, 161],
    [333, 421, 49, 49],
]


class_labels = ['cat', 'dog', 'parrot']

'''
然后，您将两个边界框和类别标签传递给 transform 。请注意，要传递类别标签，您需要使用在步骤 2 中
创建 Compose 实例时声明的参数名称。在我们的例子中，我们将参数名称设置为 class_labels 。
'''



transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
transformed_image = transformed['image']
transformed_bboxes = transformed['bboxes']
transformed_class_labels = transformed['class_labels']


transform = A.Compose([
    A.RandomCrop(width=450, height=450),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels', 'class_categories'])))



class_labels = ['cat', 'dog', 'parrot']
class_categories = ['animal', 'animal', 'item']

transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels, class_categories=class_categories)
transformed_image = transformed['image']
transformed_bboxes = transformed['bboxes']
transformed_class_labels = transformed['class_labels']
transformed_class_categories = transformed['class_categories']










