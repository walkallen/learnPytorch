

'''
Reference
    https://huggingface.co/docs/transformers/v4.48.2/zh/training


在原生 PyTorch 中训练



Trainer 负责训练循环, 允许您在一行代码中微调模型。对于喜欢编写自己训练循环的用户, 您也可以在原生 PyTorch 中微调 🤗 Transformers 模型。

现在, 您可能需要重新启动您的notebook, 或执行以下代码以释放一些内存：


'''

from datasets import load_dataset

dataset = load_dataset("yelp_review_full")

print('\n')
print('评论数据集')
print(
dataset
)

print('\n')

print(
dataset["train"][100]
)


# 正如您现在所知, 您需要一个tokenizer来处理文本, 包括填充和截断操作以处理可变的序列长度。
# 如果要一次性处理您的数据集, 可以使用 🤗 Datasets 的 map 方法, 将预处理函数应用于整个数据集：



print('\n')
print('您需要一个tokenizer来处理文本')


from transformers import AutoTokenizer



tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")




def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)

print(
    tokenized_datasets
)


'''


接下来，手动处理 tokenized_dataset 以准备进行训练。



'''

# 1. 移除 text 列，因为模型不接受原始文本作为输入：


print('\n')
print('1. 移除 text 列，因为模型不接受原始文本作为输入：')

tokenized_datasets = tokenized_datasets.remove_columns(["text"])



print('2. 将 label 列重命名为 labels，因为模型期望参数的名称为 labels：')

# 2. 将 label 列重命名为 labels，因为模型期望参数的名称为 labels：

tokenized_datasets = tokenized_datasets.rename_column("label", "labels")


# 3. 设置数据集的格式以返回 PyTorch 张量而不是lists：

print('3. 设置数据集的格式以返回 PyTorch 张量而不是lists：')

tokenized_datasets.set_format("torch")

print('\n')
print(
    tokenized_datasets
)


# 接着，创建一个先前展示的数据集的较小子集，以加速微调过程

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))



'''
DataLoader
您的训练和测试数据集创建一个DataLoader类，以便可以迭代处理数据批次


'''


from torch.utils.data import DataLoader

train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=8)
eval_dataloader  = DataLoader(small_eval_dataset, batch_size=8)



# 加载您的模型，并指定期望的标签数量：



from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=5)


# Optimizer and learning rate scheduler
# 创建一个 optimizer 和 learning rate scheduler 以进行模型微调。让我们使用 PyTorch 中的 AdamW 优化器：


from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5)


# 创建来自 Trainer 的默认 learning rate scheduler：

from transformers import get_scheduler

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)


# 最后，指定 device 以使用 GPU（如果有的话）。否则，使用 CPU 进行训练可能需要几个小时，而不是几分钟。


import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)



import evaluate

metric = evaluate.load("accuracy")
model.eval()                        # 评估模式，关闭 Dropout 层
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():           # 临时禁用梯度计算，令 torch 不记录计算图，可以减少内存消耗
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)  # 将 logits 转换为模型预测的类别 (predictions)。 
                                                # 如果 logits 是形状为 (batch_size, num_classes) 的张量， 
                                                # 那么 predictions 将会是形状为 (batch_size) 的张量， 
                                                # 其中每个元素代表对应样本的预测类别索引。
    metric.add_batch(predictions=predictions, references=batch["labels"])

print(
metric.compute()
)




# 现在您已经准备好训练了！🥳

# 训练循环
# 为了跟踪训练进度，使用 tqdm 库来添加一个进度条，显示训练步数的进展：



from tqdm.auto import tqdm

progress_bar = tqdm(range(num_training_steps))

model.train() # 训练模式，使用 Dropout 层
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)# 解包字典 如果 batch 包含键 "input_ids" 和 "attention_mask"， 那么这行代码就相当于 outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        loss = outputs.loss     # 从模型的输出 outputs 中提取损失值 (loss)
        loss.backward()         # 执行反向传播 (backpropagation)， 计算模型参数的梯度 (gradients)。

        optimizer.step()        # 使用优化器 (optimizer) 来更新模型的参数。
        lr_scheduler.step()     # 调用学习率调度器 (learning rate scheduler) 来更新学习率 (learning rate)
        optimizer.zero_grad()   # 将优化器中存储的梯度清零。
        progress_bar.update(1)  # 更新进度条 (progress bar)， 用于显示训练进度





import evaluate

metric = evaluate.load("accuracy")
model.eval()                        # 评估模式，关闭 Dropout 层
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():           # 临时禁用梯度计算，令 torch 不记录计算图，可以减少内存消耗
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)  # 将 logits 转换为模型预测的类别 (predictions)。 
                                                # 如果 logits 是形状为 (batch_size, num_classes) 的张量， 
                                                # 那么 predictions 将会是形状为 (batch_size) 的张量， 
                                                # 其中每个元素代表对应样本的预测类别索引。
    metric.add_batch(predictions=predictions, references=batch["labels"])

print(
metric.compute()
)


