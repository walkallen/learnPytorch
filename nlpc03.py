
'''
Reference

https://huggingface.co/learn/nlp-course/chapter3/4?fw=pt


A full training  全面培训

现在我们将看到如何在不使用 Trainer 类的情况下实现与上一节相同的结果。
再次假设你已经完成了第 2 节中的数据处理。以下是涵盖你需要的一切的简要总结：




'''

from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


'''
准备训练


在实际上编写我们的训练循环之前，我们需要定义一些对象。首先是我们将用于遍历批次的加载数据器。
但在定义这些加载数据器之前，我们需要对我们的 tokenized_datasets 进行一些后处理，
以处理 Trainer 为我们自动处理的一些事情。具体来说，我们需要：

删除模型不期望的值对应的列（如 sentence1 和 sentence2 列）。

重命名列 label 为 labels （因为模型期望参数名为 labels ）。

设置数据集的格式，以便它们返回 PyTorch 张量而不是列表。

我们的 tokenized_datasets 为每个步骤提供了一个方法：

'''

tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

print(
tokenized_datasets["train"].column_names
)

'''
我们可以检查结果只包含模型可以接受的列：

["attention_mask", "input_ids", "labels", "token_type_ids"]

现在这完成了，我们可以轻松定义我们的数据加载器：


'''

from torch.utils.data import DataLoader

train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
)

'''
为了快速检查数据处理中是否存在错误，我们可以检查这样的一个批次：

'''

for batch in train_dataloader:
    break

print(
{k: v.shape for k, v in batch.items()}
)
print("\n")

'''
{
    'labels': torch.Size([8]), 
    'input_ids': torch.Size([8, 66]), 
    'token_type_ids': torch.Size([8, 66]), 
    'attention_mask': torch.Size([8, 66])
}

请注意，实际形状可能因您而异，因为我们为训练数据加载器设置了 shuffle=True ，并且在批次内部填充到最大长度。

现在我们已经完全完成了数据预处理（对于任何机器学习从业者来说，这是一个令人满意但难以捉摸的目标），
让我们转向模型。我们将其实例化，就像在上一节中做的那样：


'''

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

'''
为确保训练过程中一切顺利，我们将我们的批次传递给此模型：


'''

outputs = model(**batch)
print(outputs.loss, outputs.logits.shape)
print("\n")

'''
tensor(0.6937, grad_fn=<NllLossBackward0>) torch.Size([8, 2])


所有🤗 Transformer 模型在提供 labels 时将返回损失，我们还得到 logits (每个输入两个，因此是一个 8 x 2 大小的张量）。

我们几乎准备好编写训练循环了！我们只缺两样东西：一个优化器和学习率调度器。
由于我们正在尝试手动复制 Trainer 所做的事情，我们将使用相同的默认设置。 
Trainer 使用的优化器是 AdamW ，它与 Adam 相同，但为了权重衰减正则化有所调整
（参见 Ilya Loshchilov 和 Frank Hutter 的“解耦权重衰减正则化”）：
'''

from transformers import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5)



'''

最后, 默认使用的学习率调度器只是一个从最大值 (5e-5) 到 0 的线性衰减。
为了正确定义它，我们需要知道我们将要进行的训练步数，
这等于我们想要运行的 epoch 数乘以训练批次数（这是我们的训练数据加载器的长度）。 
Trainer 默认使用三个 epoch, 因此我们将遵循这个设置：

'''


from transformers import get_scheduler

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
print(num_training_steps)
print("\n")



'''
The training loop  训练循环

最后一件事：如果我们能访问到 GPU, 我们会想使用它 (在 CPU 上，训练可能需要几个小时而不是几分钟）。
为此，我们定义了一个 device ，我们将把我们的模型和批次放在上面：


'''


import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
print(device)
print('\n')



'''
我们现在准备开始训练！为了了解训练何时结束，我们在训练步数上添加了一个进度条，使用的是 tqdm 库：

'''


from tqdm.auto import tqdm

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)




'''
The evaluation loop  评估循环

我们之前已经这样做过了，我们将使用🤗 Evaluate 库提供的指标。
我们已经看到了 metric.compute() 方法，但实际上，指标可以在我们使用 add_batch() 
方法遍历预测循环时为我们累积批次。一旦我们累积了所有批次，
我们就可以使用 metric.compute() 获取最终结果。
以下是如何在评估循环中实现所有这些内容的示例：


'''


import evaluate

metric = evaluate.load("glue", "mrpc")
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

print(
metric.compute()
)















