
from datasets import load_dataset

print("start load datasets")
raw_datasets = load_dataset("glue", "mrpc")
print(raw_datasets)

'''
DatasetDict 词典，内部有 3 个 dataset1

131

DatasetDict({
    train: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 3668
    })
    validation: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 408
    })
    test: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 1725
    })
})

'''

raw_train_dataset = raw_datasets["train"]
print(raw_train_dataset[0])

'''
判断第一句话和第二句话是否相关

{
    'sentence1': 'Amrozi accused his brother , whom he called " the witness " , of deliberately distorting his evidence .', 
    'sentence2': 'Referring to him as only " the witness " , Amrozi accused his brother of deliberately distorting his evidence .', 
    'label': 1, 
    'idx': 0
}

BERT 分词器需要同时处理两句话, BERT 分词器支持这个功能
'''

from transformers import AutoTokenizer

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenized_sentences_1 = tokenizer(raw_datasets["train"]["sentence1"])
tokenized_sentences_2 = tokenizer(raw_datasets["train"]["sentence2"])


inputs = tokenizer("This is the first sentence.", "This is the second one.")
print(inputs)


'''
{
    'input_ids': [101, 2023, 2003, 1996, 2034, 6251, 1012, 102, 2023, 2003, 1996, 2117, 2028, 1012, 102], 
    'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1], 
    'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
}


至此我们可以将整个数据集转换为 token 格式

tokenized_dataset = tokenizer(
    raw_datasets["train"]["sentence1"],
    raw_datasets["train"]["sentence2"],
    padding=True,
    truncation=True,
)

但是内存有限，一次性转换过亿数据，机器的内存不够，所以我们得构造一个 dataloader, 按照批次转换

我们将使用 Dataset.map() 方法。这也为我们提供了一些额外的灵活性，
如果我们需要做的预处理不仅仅是分词。 map() 方法通过在数据集的每个元素上应用一个函数来实现，所以让我们定义一个函数来分词我们的输入：



'''

def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


'''
此函数接收一个字典（如我们的数据集项）并返回一个新的字典，其键为 input_ids 、 attention_mask 和 token_type_ids 。
注意，如果 example 字典包含多个样本（每个键为句子列表），它也可以正常工作，因为 tokenizer 在句子对的列表上工作，
正如之前所见。这将允许我们在调用 map() 时使用 batched=True 选项，这将大大加快分词速度。
 tokenizer 由 Rust 编写的分词器支持，该分词器可以非常快，但前提是我们一次给它提供大量输入。

注意，我们目前省略了 tokenization 函数中的 padding 参数。这是因为将所有样本填充到最大长度并不高效：
我们最好在构建批次时填充样本，这样我们只需要填充到该批次的最大长度，而不是整个数据集的最大长度。
当输入长度非常不固定时，这可以节省大量时间和处理能力！

这里是我们一次性将标记化函数应用于所有数据集的方法。我们在调用 map 时使用 batched=True ，
这样函数就可以一次性应用于数据集的多个元素，而不是单独应用于每个元素。这允许更快地进行预处理。


'''

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
print(tokenized_datasets)


'''

原来的数据集是如下几列，['sentence1', 'sentence2', 'label', 'idx'], 经过 map 操作以后，增加了而 tokennize 之后的几列

DatasetDict({
    train: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx', 'input_ids', 'token_type_ids', 'attention_mask'],
        num_rows: 3668
    })
    validation: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx', 'input_ids', 'token_type_ids', 'attention_mask'],
        num_rows: 408
    })
    test: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx', 'input_ids', 'token_type_ids', 'attention_mask'],
        num_rows: 1725
    })
})

接下来要在 load 一个批次的时候执行 padidng

我们将需要做的最后一件事是在将元素批量组合时将所有示例填充到最长元素长度——我们称之为动态填充的技术。

该函数负责将批次内的样本组合在一起，被称为 collate 函数。这是你在构建 DataLoader 时可以传递的参数，
默认是一个将你的样本转换为 PyTorch 张量并将它们连接起来的函数（如果元素是列表、元组或字典，则递归连接）。
在我们的情况下，这是不可能的，因为我们拥有的输入不会全部具有相同的大小。我们故意推迟填充，
仅在必要时对每个批次应用填充，以避免有大量填充的过长输入。这将大大加快训练速度，但请注意，如果你在 TPU 上训练，
可能会引起问题——TPU 更喜欢固定形状，即使那需要额外的填充。

'''

from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


samples = tokenized_datasets["train"][:8]
samples = {k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]}


batch = data_collator(samples)

print({k: v.shape for k, v in batch.items()})

'''
{
    'input_ids': torch.Size([8, 67]), 
    'token_type_ids': torch.Size([8, 67]), 
    'attention_mask': torch.Size([8, 67]), 
    'labels': torch.Size([8])
}

'''