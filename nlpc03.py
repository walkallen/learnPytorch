
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

'''
{
    'labels': torch.Size([8]), 
    'input_ids': torch.Size([8, 66]), 
    'token_type_ids': torch.Size([8, 66]), 
    'attention_mask': torch.Size([8, 66])
}

请注意，实际形状可能因您而异，因为我们为训练数据加载器设置了 shuffle=True ，并且在批次内部填充到最大长度。

现在我们已经完全完成了数据预处理（对于任何机器学习从业者来说，这是一个令人满意但难以捉摸的目标），让我们转向模型。我们将其实例化，就像在上一节中做的那样：


'''

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)









