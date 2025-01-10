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


# trainer.train()


'''

{'loss': 0.4866, 'grad_norm': 6.259756565093994, 'learning_rate': 3.184458968772695e-05, 'epoch': 1.09}                                           
{'loss': 0.236, 'grad_norm': 0.13721801340579987, 'learning_rate': 1.3689179375453886e-05, 'epoch': 2.18}                                         
{'train_runtime': 83.7762, 'train_samples_per_second': 131.35, 'train_steps_per_second': 16.437, 'train_loss': 0.291377137515886, 'epoch': 3.0} 


这将开始微调（在 GPU 上应该需要几分钟时间）并每 500 步报告一次训练损失。然而，它不会告诉你你的模型表现如何（好或坏）。这是因为：

1. 我们没有通过将 evaluation_strategy 设置为 "steps" （每 eval_steps 评估一次）或 "epoch" 
    （在每个 epoch 结束时评估）来告诉 Trainer 在训练期间进行评估。

2. 我们没有为 Trainer 提供 compute_metrics() 功能来在所述评估期间计算一个指标（否则评估只会打印损失，这不是一个很直观的数字）。


让我们看看我们如何构建一个有用的 compute_metrics() 函数，并在下次训练时使用它。
该函数必须接受一个 EvalPrediction 对象（它是一个具有 predictions 字段和 label_ids 字段的命名元组）
并返回一个将字符串映射到浮点数的字典（字符串是返回的指标名称，浮点数是它们的值）。
要从我们的模型获取一些预测，我们可以使用 Trainer.predict() 命令：



'''

predictions = trainer.predict(tokenized_datasets["validation"])
print(predictions.predictions.shape, predictions.label_ids.shape)



'''

predict() 方法的输出是另一个包含三个字段的命名元组： predictions 、 label_ids 和 metrics 。 
metrics 字段将仅包含传递给数据集的损失以及一些时间指标（预测所需的总时间和平均时间）。
一旦我们完成我们的 compute_metrics() 函数并将其传递给 Trainer ，该字段也将包含 compute_metrics() 返回的指标。

如您所见， predictions 是一个形状为 408 x 2 的二维数组 (408 是我们在数据集中使用的元素数量）。
这些是传递给 predict() 的每个数据集元素的 logits (正如您在上一章中看到的，所有 Transformer 模型都返回 logits) 。要将它们转换为可以与我们标签进行比较的预测，我们需要在第二个轴上取最大值的索引：

'''

import numpy as np

preds = np.argmax(predictions.predictions, axis=-1)



'''

我们现在可以将这些 preds 与标签进行比较。为了构建我们的 compute_metric() 函数，
我们将依赖于🤗 Evaluate 库中的指标。我们可以像加载数据集一样轻松地加载与 MRPC 数据集相关的指标，
这次使用 evaluate.load() 函数。返回的对象有一个我们可以用来进行指标计算的 compute() 方法：

'''



import evaluate

metric = evaluate.load("glue", "mrpc")

print(
metric.compute(predictions=preds, references=predictions.label_ids)

)


'''

您得到的精确结果可能会有所不同，因为模型头的随机初始化可能会改变其达到的指标。
在这里，我们可以看到我们的模型在验证集上的准确率为 85.78%，F1 分数为 89.97。
这两个指标用于评估 GLUE 基准测试中 MRPC 数据集上的结果。BERT 论文中的表格报告了
基础模型的 F1 分数为 88.9。那是 uncased 模型，而我们目前使用的是 cased 模型，这解释了更好的结果。

将所有内容整合在一起，我们得到我们的 compute_metrics() 函数：

'''

def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


'''
为了看到它在每个 epoch 结束时报告指标的应用，以下是使用此 compute_metrics() 函数定义新 Trainer 的方法：

'''

training_args = TrainingArguments("test-trainer", evaluation_strategy="epoch")
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()



'''
这次，它将在每个 epoch 结束时报告验证损失和指标，除了训练损失。
再次强调，您达到的精确准确率/F1 分数可能与我们所找到的略有不同，因为模型随机头初始化的原因，但它应该在大致相同的范围内。


该 Trainer 在多个 GPU 或 TPU 上即插即用，并提供大量选项，
如混合精度训练（在您的训练参数中使用 fp16 = True ）。我们将在第 10 章中详细介绍它支持的所有功能。

本节介绍了使用 Trainer API 进行微调的入门。第 7 章将给出一个针对大多数常见 NLP 任务的示例，
但在此我们先看看如何在纯 PyTorch 中实现相同的功能。


'''







