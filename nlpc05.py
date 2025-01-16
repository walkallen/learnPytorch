
'''
Reference

https://huggingface.co/learn/nlp-course/chapter4/4?fw=pt


构建模型卡片


模型卡片是一个文件，它可能和模型以及分词器文件一样重要，在模型仓库中。
它是模型的中心定义，确保社区成员的可重用性和结果的可重复性，并为其他成员构建他们的工件提供了一个平台。

记录训练和评估过程有助于他人了解对模型可以期待什么——提供有关所使用数据
以及所进行的预处理和后处理足够的信息，确保可以识别和理解模型有用和无效的限制、偏差和情境。

因此，创建一个清晰定义您模型的模型卡是一个非常重要的步骤。
在此，我们提供一些有助于您的建议。创建模型卡是通过您之前看到的 README.md 文件完成的，这是一个 Markdown 文件

“模型卡”概念源自谷歌的一个研究方向，首次在 Margaret Mitchell 等人撰写的《模型报告的模型卡》
一文中提出。这里包含的大量信息基于该论文，我们建议您阅读它以了解为什么在重视可重复性、可重用性和公平性的世界中，模型卡如此重要。


模型卡片通常以非常简短、高级的概述开始，说明该模型的作用，
随后在以下章节中提供更多详细信息：

Model description           模型描述
Intended uses & limitations 预期用途和限制
How to use                  如何使用
Limitations and bias        局限性及偏见
Training data               训练数据
Training procedure          训练过程
Evaluation results          评估结果




Model description           模型描述

模型描述提供了关于模型的基本细节。这包括架构、版本、是否在论文中介绍、是否有原始实现、
作者以及关于模型的一般信息。任何版权都应该在这里归属。关于训练过程、参数和重要免责声明的一般信息也可以在本节中提及。



Intended uses & limitations 预期用途和限制

这里描述了模型旨在使用的用例，包括它可以应用的语言、领域和领域，以及模型已知范围之外或可能表现不佳的领域。


How to use                  如何使用

本节应包含一些如何使用该模型的示例。这可以展示 pipeline() 函数的使用、模型和分词器类的使用，以及任何你认为可能有帮助的代码。


Training data               训练数据

这部分应指明模型是在哪些数据集上训练的。也欢迎提供数据集的简要描述。


Training procedure          训练过程

本节中，您应描述所有从可重复性角度有用的训练相关方面。这包括对数据进行的所有预处理和后处理，
以及诸如模型训练的 epoch 数、批量大小、学习率等细节。



Variable and metrics        变量和指标

这里应描述您用于评估的指标以及您正在测量的不同因素。提及使用了哪些指标、
在哪个数据集以及哪个数据集分割上，可以轻松比较您模型的表现与其他模型的表现。这些应受先前章节的指导，例如目标用户和使用案例。



Evaluation results          评估结果

最后，提供模型在评估数据集上的表现指标。如果模型使用决策阈值，
则提供在评估中使用的决策阈值，或提供针对预期用途在不同阈值下的评估细节。


Note  注意

模型卡片不是发布模型的要求，您在制作时不需要包含上述所有部分。然而，
对模型的明确文档化只会对未来的用户有益，所以我们建议您尽可能根据您的知识和能力填写尽可能多的部分。




第三章中，你第一次接触到了🤗 Datasets 库，并了解到微调模型时主要有三个步骤：


1. 从 Hugging Face Hub 加载一个数据集。
2. 预处理数据使用 Dataset.map() 。
3. 加载并计算指标。


但这只是触及了🤗 Datasets 能做什么的皮毛！在本章中，我们将深入探讨这个库。在这个过程中，我们将找到以下问题的答案：



当你数据集不在 Hub 上时，你该怎么办？

如何切片和切块数据集？（如果你真的需要使用 Pandas 呢？）

当你的数据集很大，会熔化你的笔记本电脑的 RAM 时，你该怎么办？

什么鬼是“内存映射”和 Apache Arrow？


如何创建自己的数据集并将其推送到 Hub？



datasets 模块提供加载脚本以处理本地和远程数据集的加载。它支持多种常见数据格式，例如：

load_dataset("csv", data_files="my_file.csv")
load_dataset("text", data_files="my_file.txt")
load_dataset("json", data_files="my_file.jsonl")
load_dataset("pandas", data_files="my_dataframe.pkl")


如表所示，对于每种数据格式，我们只需在 load_dataset() 函数中指定加载脚本的类型，
以及一个 data_files 参数，用于指定一个或多个文件的路径。
让我们先从本地文件加载数据集开始；稍后我们将看到如何使用远程文件进行相同的操作。



Loading a local dataset  加载本地数据集


对于这个例子，我们将使用 SQuAD-it 数据集，这是一个用于意大利语问答的大规模数据集。

wget https://github.com/crux82/squad-it/raw/master/SQuAD_it-train.json.gz
wget https://github.com/crux82/squad-it/raw/master/SQuAD_it-test.json.gz
gzip -dkv SQuAD_it-*.json.gz


加载一个 JSON 文件使用 load_dataset() 函数，我们只需要知道我们处理的是普通
 JSON（类似于嵌套字典）还是 JSON Lines（行分隔的 JSON）。像许多问答数据集一样，
 SQuAD-it 使用嵌套格式，所有文本都存储在 data 字段中。这意味着我们可以通过指定以下 field 参数来加载数据集：


'''


from datasets import load_dataset

squad_it_dataset = load_dataset("json", data_files="SQuAD_it-train.json", field="data")


print(squad_it_dataset)

'''
这显示了与训练集相关的行数和列名。我们可以通过以下方式通过索引到 train 分割来查看其中一个示例：


DatasetDict({
    train: Dataset({
        features: ['title', 'paragraphs'],
        num_rows: 442
    })
})


'''


# print(squad_it_dataset["train"][0])

'''
{
    "title": "Terremoto del Sichuan del 2008",
    "paragraphs": [
        {
            "context": "Il terremoto del Sichuan del 2008 o il terremoto...",
            "qas": [
                {
                    "answers": [{"answer_start": 29, "text": "2008"}],
                    "id": "56cdca7862d2951400fa6826",
                    "question": "In quale anno si è verificato il terremoto nel Sichuan?",
                },
                ...
            ],
        },
        ...
    ],
}


太好了，我们已经加载了我们的第一个本地数据集！但是，虽然这对训练集有效，
我们真正想要的是将 train 和 test 分割包含在一个单一的 DatasetDict 对象中，
这样我们就可以一次性对两个分割应用 Dataset.map() 函数。
为此，我们可以向 data_files 参数提供一个字典，将每个分割名称映射到与该分割关联的文件：



'''


data_files = {"train": "SQuAD_it-train.json", "test": "SQuAD_it-test.json"}
squad_it_dataset = load_dataset("json", data_files=data_files, field="data")
print(squad_it_dataset)















