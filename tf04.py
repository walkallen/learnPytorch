



'''

Reference
    https://huggingface.co/docs/transformers/v4.48.2/zh/autoclass_tutorial


使用AutoClass加载预训练实例
    由于存在许多不同的 Transformer 架构, 因此为您的 checkpoint 创建一个可用架构可能会具有挑战性。
    通过 AutoClass 可以自动推断并从给定的 checkpoint 加载正确的架构, 这也是🤗 Transformers
    易于使用、简单且灵活核心规则的重要一部分。from_pretrained() 方法允许您快速加载任何架构的
    预训练模型, 因此您不必花费时间和精力从头开始训练模型。生成这种与checkpoint无关的代码意味着, 
    如果您的代码适用于一个checkpoint, 它将适用于另一个checkpoint - 只要它们是为了类似的
    任务进行训练的 - 即使架构不同。

请记住, 架构指的是模型的结构, 而 checkpoints 是给定架构的权重。例如, 
    BERT是一种架构, 而 google-bert/bert-base-uncased 是一个 checkpoint。
    模型是一个通用术语, 可以指代架构或 checkpoint 。


在这个教程中, 学习如何：

    加载预训练的分词器（tokenizer）
    加载预训练的图像处理器(image processor)
    加载预训练的特征提取器(feature extractor)
    加载预训练的处理器(processor)
    加载预训练的模型。


AutoTokenizer

    几乎所有的NLP任务都以 tokenizer 开始。tokenizer 将您的输入转换为模型可以处理的格式。
    使用 AutoTokenizer.from_pretrained() 加载 tokenizer：


'''


from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

# 然后按照如下方式对输入进行分词：


sequence = "In a hole in the ground there lived a hobbit."
print(tokenizer(sequence))

'''
{
    'input_ids': [101, 1999, 1037, 4920, 1999, 1996, 2598, 2045, 2973, 1037, 7570, 10322, 4183, 1012, 102], 
    'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
}

AutoImageProcessor
    
    对于视觉任务, image processor将图像处理成正确的输入格式。


'''

from transformers import AutoImageProcessor

image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")


'''
AutoFeatureExtractor
    
    对于音频任务,feature extractor将音频信号处理成正确的输入格式。

    使用 AutoFeatureExtractor.from_pretrained() 加载 feature extractor:


'''

from transformers import AutoFeatureExtractor

feature_extractor = AutoFeatureExtractor.from_pretrained(
    "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
)


'''
AutoProcessor
    多模态任务需要一种 processor, 将两种类型的预处理工具结合起来。例如, LayoutLMV2
    模型需要一个 image processor 来处理图像和一个 tokenizer 来处理文本; processor 将两者结合起来。

使用 AutoProcessor.from_pretrained() 加载 processor:


'''




from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("microsoft/layoutlmv2-base-uncased")




'''
AutoModel

    最后, AutoModelFor类让你可以加载给定任务的预训练模型（参见这里获取可用任务的完整列表）。
    例如, 使用AutoModelForSequenceClassification.from_pretrained()加载用于序列分类的模型：


'''


from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased")


# 轻松地重复使用相同的 checkpoint 来为不同任务加载模型架构：

from transformers import AutoModelForTokenClassification

model = AutoModelForTokenClassification.from_pretrained("distilbert/distilbert-base-uncased")


'''
    一般来说，我们建议使用AutoTokenizer类和AutoModelFor类来加载预训练的模型实例。
    这样可以确保每次加载正确的架构。在下一个教程中，学习如何使用新加载的tokenizer, image processor, 
    feature extractor和processor对数据集进行预处理以进行微调。



'''















