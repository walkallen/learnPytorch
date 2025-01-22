import requests

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 

from datasets import load_from_disk


model_id = "IDEA-Research/grounding-dino-tiny"
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)


loaded_dataset_dict = load_from_disk('/mnt/SD1T/suhao/learnPytorchbak/learnPytorch/data_pku_msd')

print('data pku msd train features')
print(loaded_dataset_dict['train'].features)
print('\n')

'''
{
    'image': Image(mode=None, decode=True, id=None), 
    'image_id': Value(dtype='int64', id=None), 
    'file_name': Value(dtype='string', id=None), 
    'width': Value(dtype='int64', id=None), 
    'height': Value(dtype='int64', id=None), 
    'objects': 
        {'bbox': 
            Sequence(feature=Sequence(feature=Value(dtype='float64', id=None), length=-1, id=None), 
            length=-1, id=None), 
            'categories': Sequence(feature=Value(dtype='int64', id=None), length=-1, id=None)
        }
}

'''
print(loaded_dataset_dict['train'].features['objects'])
print('\n')


# image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(image_url, stream=True).raw)

# image = loaded_dataset_dict['test'][0]['image']
image = loaded_dataset_dict['test'][82]['image']


print(image)
image.show()

# Check for cats and remote controls
# VERY important: text queries need to be lowercased + end with a dot
# text = "a cat. a remote control."
text = "a oil. a phone. a scratch."
# text = " oil. phone. scratch."

inputs = processor(images=image, text=text, return_tensors="pt").to(device)


print(inputs)
print("\n")

with torch.no_grad():
    outputs = model(**inputs)


# sbox_threshold = 0.1
# stext_threshold = 0.1

sbox_threshold = 0.2
stext_threshold = 0.2

# sbox_threshold = 0.4
# stext_threshold = 0.3

results = processor.post_process_grounded_object_detection(
    outputs,
    inputs.input_ids,
    box_threshold   = sbox_threshold,
    text_threshold  = stext_threshold,
    target_sizes    = [image.size[::-1]]
)

result_first = results[0]

print(results)


from PIL import ImageDraw 
draw = ImageDraw.Draw(image)
from PIL import Image, ImageDraw, ImageFont
font = ImageFont.load_default()
font.size = 30


# 绘制推理得到的结果
for score, label, box in zip(result_first["scores"], result_first["labels"], result_first["boxes"]):
    # 四舍五入，限制小数点两位
    box = [round(i, 2) for i in box.tolist()]
    print(
        f"Detected {label} with confidence "
        f"{round(score.item(), 3)} at location {box}"
    )

    # 左上角，右下角，两个点
    x, y, x2, y2 = tuple(box)
    draw.rectangle((x, y, x2, y2), outline="blue", width=1)
    draw.text((x, y), label, fill="blue", font=font)


# 绘制标注的标签



image.show()