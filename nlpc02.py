'''

https://huggingface.co/learn/nlp-course/chapter3/3?fw=pt


Transformers 提供了一个 Trainer 类来帮助您在您的数据集上微调它提供的任何预训练模型。
一旦您在上一个部分完成所有数据预处理工作，您只需定义 Trainer 的几个步骤。
最困难的部分可能是准备运行 Trainer.train() 的环境，因为它在 CPU 上运行会非常慢。
如果您没有设置 GPU，您可以在 Google Colab 上获取免费的 GPU 或 TPU。


以下代码示例假设您已经执行了上一节中的示例。这里是一个简短的总结，回顾您需要的内容：
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

首先，在我们可以定义我们的 Trainer 之前，需要定义一个 TrainingArguments 类，
该类将包含 Trainer 在训练和评估过程中将使用的所有超参数。您必须提供的唯一参数是一个目录，
其中将保存训练好的模型以及过程中的检查点。对于其余部分，您可以保留默认设置，这对于基本的微调应该相当不错。


'''

from transformers import TrainingArguments

training_args = TrainingArguments("test-trainer")


# 第二步是定义我们的模型。与上一章一样，我们将使用 AutoModelForSequenceClassification 类，有两个标签：

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

'''
您将注意到，与第二章不同，在实例化此预训练模型后，您会收到一个警告。
这是因为 BERT 尚未在分类句子对方面进行预训练，因此预训练模型的头部已被丢弃，并添加了一个适合序列分类的新头部。
警告表明，一些权重未被使用（对应于丢弃的预训练头部）以及一些其他权重被随机初始化（对应于新头部）。
最后，它鼓励您训练模型，这正是我们现在要做的。

Some weights of BertForSequenceClassification were not initialized from the model checkpoint 
at bert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight']

You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.



一旦我们有了我们的模型，我们可以通过传递到目前为止构建的所有对象来定义一个 Trainer —— 包括 model 、 
training_args 、训练和验证数据集、我们的 data_collator 和 tokenizer ：


'''


from transformers import Trainer

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

'''
请注意，当您像这里一样传递 tokenizer 时， Trainer 使用的默认 data_collator 将是一个之前定义的 DataCollatorWithPadding ，
因此您可以跳过此调用中的 data_collator=data_collator 行。在第二部分中展示这一处理部分仍然很重要！

为了在我们的数据集上微调模型，我们只需调用我们的 Trainer 的 train() 方法即可


'''


trainer.train()


'''
这将开始微调（在 GPU 上应该需要几分钟时间）并每 500 步报告一次训练损失。然而，它不会告诉你你的模型表现如何（好或坏）。这是因为：



'''