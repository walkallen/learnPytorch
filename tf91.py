

'''
Reference



'''

from transformers import AutoImageProcessor, ConditionalDetrForObjectDetection
import torch
from PIL import Image
import requests

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

processor = AutoImageProcessor.from_pretrained("microsoft/conditional-detr-resnet-50")
model = ConditionalDetrForObjectDetection.from_pretrained("microsoft/conditional-detr-resnet-50")

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)

print(image.size)

# convert outputs (bounding boxes and class logits) to COCO API
# let's only keep detections with score > 0.7
target_sizes = torch.tensor([image.size[::-1]])

print(target_sizes)

results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.7)[0]

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    print(
            f"Detected {model.config.id2label[label.item()]} with confidence "
            f"{round(score.item(), 3)} at location {box}"
    )



