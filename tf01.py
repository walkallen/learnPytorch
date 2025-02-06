'''

Reference:
    https://huggingface.co/docs/transformers/v4.48.2/zh/quicktour


文本分类	    为给定的文本序列分配一个标签	                        NLP	     pipeline(task=“sentiment-analysis”)
文本生成	    根据给定的提示生成文本	                               NLP	    ipeline(task=“text-generation”)
命名实体识别	 为序列里的每个 token 分配一个标签（人, 组织, 地址等等）    NLP 	pipeline(task=“ner”)
问答系统	    通过给定的上下文和问题, 在文本中提取答案	              NLP	  pipeline(task=“question-answering”)
掩盖填充	    预测出正确的在序列中被掩盖的token	                    NLP	    pipeline(task=“fill-mask”)
文本摘要	    为文本序列或文档生成总结	                           NLP	   pipeline(task=“summarization”)
文本翻译	    将文本从一种语言翻译为另一种语言	                    NLP	     pipeline(task=“translation”)


图像分类	    为图像分配一个标签	                                 Computer vision	  pipeline(task=“image-classification”)
图像分割	    为图像中每个独立的像素分配标签（支持语义、全景和实例分割）	Computer vision	    pipeline(task=“image-segmentation”)
目标检测	    预测图像中目标对象的边界框和类别	                    Computer vision	    pipeline(task=“object-detection”)


音频分类	    给音频文件分配一个标签	                              Audio	               pipeline(task=“audio-classification”)
自动语音识别	将音频文件中的语音提取为文本	                        Audio	pipeline(task=“automatic-speech-recognition”)

视觉问答	给定一个图像和一个问题，正确地回答有关图像的问题	Multimodal	pipeline(task=“vqa”)


suhao, 可以理解为所有上传到 hugging face 的模型都实现的 huggingface 的 api 接口，比如 pipeline 等等
'''

from transformers import pipeline

classifier = pipeline("sentiment-analysis")

print(
    classifier("We are very happy to show you the 🤗 Transformers library.")
)


'''
如果你有不止一个输入，可以把所有输入放入一个列表然后传给pipeline()，它将会返回一个字典列表：

'''

print('如果你有不止一个输入，可以把所有输入放入一个列表然后传给pipeline()，它将会返回一个字典列表：')
results = classifier(["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it."])
for result in results:
    print(f"label: {result['label']}, with score: {round(result['score'], 4)}")




'''
在 pipeline 中使用另一个模型和分词器
    pipeline() 可以容纳 Hub 中的任何模型，这让 pipeline() 更容易适用于其他用例。
    比如，你想要一个能够处理法语文本的模型，就可以使用 Hub 上的标记来筛选出合适的模型。
    靠前的筛选结果会返回一个为情感分析微调的多语言的 BERT 模型，你可以将它用于法语文本：


使用 AutoModelForSequenceClassification 和 AutoTokenizer 来加载预训练模型和它关联的分词器
（更多信息可以参考下一节的 AutoClass）：

在 pipeline() 中指定模型和分词器，现在你就可以在法语文本上使用 classifier 了：

'''



model_name = "nlptown/bert-base-multilingual-uncased-sentiment"



from transformers import AutoTokenizer, AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 在创建 pipeline 时，指定 模型 和 分词器
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
classifier("Nous sommes très heureux de vous présenter la bibliothèque 🤗 Transformers.")
[{'label': '5 stars', 'score': 0.7273}]




'''
在幕后，是由 AutoModelForSequenceClassification 和 AutoTokenizer 一起支持你在上面用
到的 pipeline()。AutoClass 是一个能够通过预训练模型的名称或路径自动查找其架构的快捷方式。
你只需要为你的任务选择合适的 AutoClass 和它关联的预处理类。

让我们回过头来看上一节的示例，看看怎样使用 AutoClass 来重现使用 pipeline() 的结果。


AutoTokenizer
分词器负责预处理文本，将文本转换为用于输入模型的数字数组。有多个用来管理分词过程的规则，
包括如何拆分单词和在什么样的级别上拆分单词（在 分词器总结 学习更多关于分词的信息）。
要记住最重要的是你需要实例化的分词器要与模型的名称相同, 来确保和模型训练时使用相同的分词规则。

使用 AutoTokenizer 加载一个分词器:


将文本传入分词器：

分词器也可以接受列表作为输入，并填充和截断文本，返回具有统一长度的批次：

'''

from transformers import AutoTokenizer

model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)

encoding = tokenizer("We are very happy to show you the 🤗 Transformers library.")
print(encoding)

'''
{'input_ids': [101, 11312, 10320, 12495, 19308, 10114, 11391, 10855, 10103, 100, 58263, 13299, 119, 102],
 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
'''

print('分词器也可以接受列表作为输入，并填充和截断文本，返回具有统一长度的批次：')
pt_batch = tokenizer(
    ["We are very happy to show you the 🤗 Transformers library.", 
    "We hope you don't hate it."],
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt",
)

print(pt_batch)

'''
{'input_ids': tensor([[  101, 11312, 10320, 12495, 19308, 10114, 11391, 10855, 10103,   100,  58263, 13299,   119,   102],
                        [  101, 11312, 18763, 10855, 11530,   112,   162, 39487, 10197,   119, 102,     0,     0,     0]]), 
'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), 
'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]])}


🤗 Transformers 提供了一种简单统一的方式来加载预训练的实例. 
这表示你可以像加载 AutoTokenizer 一样加载 AutoModel。唯一不同的地方是为你的任务选择正确的AutoModel。
对于文本（或序列）分类，你应该加载AutoModelForSequenceClassification：



'''

from transformers import AutoModelForSequenceClassification

model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
pt_model = AutoModelForSequenceClassification.from_pretrained(model_name)

print('解包之前得到的分词结果')
# suhao 解包之前得到的分词结果
pt_outputs = pt_model(**pt_batch)

print(pt_outputs)



'''
模型在 logits 属性输出最终的激活结果. 在 logits 上应用 softmax 函数来查询概率:

'''

print('模型在 logits 属性输出最终的激活结果. 在 logits 上应用 softmax 函数来查询概率:')

from torch import nn

pt_predictions = nn.functional.softmax(pt_outputs.logits, dim=-1)
print(pt_predictions)

'''
suhao, 结果代表从 1 级到 5 级，每一个等级的概率
tensor([[0.0021, 0.0018, 0.0115, 0.2121, 0.7725],
        [0.2084, 0.1826, 0.1969, 0.1755, 0.2365]], grad_fn=<SoftmaxBackward0>)

'''

