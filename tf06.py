


'''

Reference 
    https://huggingface.co/docs/transformers/v4.48.2/zh/training

    
微调预训练模型


    使用预训练模型有许多显著的好处。它降低了计算成本, 减少了碳排放, 同时允许您使用最先进的模型, 
    而无需从头开始训练一个。🤗 Transformers 提供了涉及各种任务的成千上万的预训练模型。
    当您使用预训练模型时, 您需要在与任务相关的数据集上训练该模型。这种操作被称为微调, 
    是一种非常强大的训练技术。在本教程中, 您将使用您选择的深度学习框架来微调一个预训练模型：

使用 🤗 Transformers 的 Trainer 来微调预训练模型。
在 TensorFlow 中使用 Keras 来微调预训练模型。
在原生 PyTorch 中微调预训练模型。


准备数据集

在您进行预训练模型微调之前, 需要下载一个数据集并为训练做好准备。
之前的教程向您展示了如何处理训练数据, 现在您有机会将这些技能付诸实践！

首先, 加载Yelp评论数据集：

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



small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))




'''
训练
    此时, 您应该根据您训练所用的框架来选择对应的教程章节。
    您可以使用右侧的链接跳转到您想要的章节 



使用 PyTorch Trainer 进行训练

    🤗 Transformers 提供了一个专为训练 🤗 Transformers 模型而优化的 Trainer 类, 
    使您无需手动编写自己的训练循环步骤而更轻松地开始训练模型。Trainer API 支持各种训练选项和功能, 
    如日志记录、梯度累积和混合精度。

    首先加载您的模型并指定期望的标签数量。根据 Yelp Review 数据集卡片, 您知道有五个标签：





'''



from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=5)



'''

您将会看到一个警告, 提到一些预训练权重未被使用, 以及一些权重被随机初始化。
不用担心, 这是完全正常的！ BERT 模型的预训练 head 被丢弃, 并替换为一个随机初始化的分类head。
您将在您的序列分类任务上微调这个新模型 head, 将预训练模型的知识转移给它。


训练超参数
接下来, 创建一个 TrainingArguments 类, 其中包含您可以调整的所有超参数以及用于激活不同训练选项的标志。
对于本教程, 您可以从默认的训练超参数开始, 但随时可以尝试不同的设置以找到最佳设置。

指定保存训练检查点的位置：


'''


from transformers import TrainingArguments

training_args = TrainingArguments(output_dir="test_trainer")


'''
评估
Trainer 在训练过程中不会自动评估模型性能。您需要向 Trainer 传递一个函数来计算和展示指标。
🤗 Evaluate 库提供了一个简单的 accuracy 函数, 您可以使用 evaluate.load 函数加载它（有关更多信息, 请参阅此快速入门）：

https://huggingface.co/docs/evaluate/a_quick_tour

在 metric 上调用 compute 来计算您的预测的准确性。在将预测传递给 compute 之前, 
您需要将预测转换为logits（请记住, 所有 🤗 Transformers 模型都返回对logits）：

如果您希望在微调过程中监视评估指标, 请在您的训练参数中指定 eval_strategy 参数, 以在每个epoch结束时展示评估指标：
'''


import numpy as np
import evaluate

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(output_dir="test_trainer", eval_strategy="epoch")

'''
训练器

    创建一个包含您的模型、训练参数、训练和测试数据集以及评估函数的 Trainer 对象：
'''

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)


trainer.train()