
'''
自定义模型构建


你可以修改模型的配置类来改变模型的构建方式。配置指明了模型的属性，比如隐藏层或者注意力头的数量。
当你从自定义的配置类初始化模型时，你就开始自定义模型构建了。模型属性是随机初始化的，
你需要先训练模型，然后才能得到有意义的结果。


通过导入 AutoConfig 来开始，之后加载你想修改的预训练模型。在 AutoConfig.from_pretrained() 中，
你能够指定想要修改的属性，比如注意力头的数量：





'''

import transformers

print(f'transformers verison is {transformers.__version__}')


from transformers import AutoConfig

my_config = AutoConfig.from_pretrained("distilbert/distilbert-base-uncased", n_heads=12)


# 使用 AutoModel.from_config() 根据你的自定义配置创建一个模型：
from transformers import AutoModel

my_model = AutoModel.from_config(my_config)



'''

Trainer - PyTorch 优化训练循环
所有的模型都是标准的 torch.nn.Module，所以你可以在任何典型的训练模型中使用它们。
当你编写自己的训练循环时，🤗 Transformers 为 PyTorch 提供了一个 Trainer 类，
它包含了基础的训练循环并且为诸如分布式训练，混合精度等特性增加了额外的功能。

取决于你的任务, 你通常可以传递以下的参数给 Trainer：



'''

# 1. PreTrainedModel 或者 torch.nn.Module：


from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased")


# 2. TrainingArguments 含有你可以修改的模型超参数，比如学习率，
#    批次大小和训练时的迭代次数。如果你没有指定训练参数，那么它会使用默认值：



from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="path/to/save/folder/",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
)

# 3. 一个预处理类，比如分词器，特征提取器或者处理器：

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")


# 4. 加载一个数据集：

from datasets import load_dataset

dataset = load_dataset("rotten_tomatoes")  # doctest: +IGNORE_RESULT


# 5. 创建一个给数据集分词的函数，并且使用 map 应用到整个数据集：

def tokenize_dataset(dataset):
    return tokenizer(dataset["text"])

dataset = dataset.map(tokenize_dataset, batched=True)




# 6. 用来从数据集中创建批次的 DataCollatorWithPadding：


from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)



# 现在把所有的类传给 Trainer：

from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,
    data_collator=data_collator,
)  # doctest: +SKIP


print(trainer)

trainer.train()